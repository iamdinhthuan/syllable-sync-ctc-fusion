"""Reference-free syllable search, frozen historical-test replay and N-best export."""
import sys, os, json, argparse, itertools, time, random, math, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import train_vivos as T
import tok
from lm_model import LM, CTX
from ctc_prefix import ComponentCTC
from bank_search import bank_beam
from rescore_saved_nbest import _rank
from replay_manifest import json_safe
import gzip
import shutil
from decoding_protocol import DecodeTracker
from nbest_rescore import rescore_component_nbest
from replay_manifest import file_sha256,parse_parameters,load_frozen_config,verify_resources,dump_record,load_replay_splits

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--lm", required=True)
ap.add_argument("--beam", type=int, default=8)
ap.add_argument("--lex", default="lm", choices=["train", "lm"])
ap.add_argument("--max_sec", type=float, default=60)
ap.add_argument("--grid", default=",".join(["0:0"]+[f"{al}:{be}" for al in (0.5,0.6,0.7,0.8) for be in (2,3,4,5)]))
ap.add_argument("--tag", default="protocol_v2")
ap.add_argument("--root", default="", help="override the data root stored in result.json (other machine)")
ap.add_argument("--dev_only", action="store_true", help="grid on dev only, skip test")
ap.add_argument("--no_unk", action="store_true", help="ablation: no <unk> candidate")
ap.add_argument("--comp_w", default="1,1,1", help="weights of the initial,rhyme,tone attention log-probs")
ap.add_argument("--ctc_cw", default="1,1,1", help="ablation: weights of the initial,rhyme,tone CTC prefix scores (0 drops a component)")
ap.add_argument("--checkpoint",default="",help="explicit acoustic state dict; default RUN/best.pt")
ap.add_argument("--out_dir",default="",help="new output directory; existing artifacts are never overwritten")
ap.add_argument("--frozen_config",default="",help="development-only result JSON, required for test replay")
ap.add_argument("--export_nbest",action='store_true',help="save complete proposal scores and exact CTC values for CPU rescoring")
ap.add_argument("--limit",type=int,default=0,help="explicit smoke subset, recorded in the protocol")
ap.add_argument("--eval_root",default="",help="external test root; training vocabulary remains tied to --root")
ap.add_argument("--evaluation_manifest",default="",help="precommitted external-evaluation description")
ap.add_argument('--completed_bank',action='store_true')
ap.add_argument('--bank_size',type=int,default=32)
ap.add_argument('--end_window',type=int,default=3)
ap.add_argument('--end_threshold',type=float,default=-10.)
ap.add_argument('--target_full',default='',help='Unchanged Full development record for exact completed target scoring')
a = ap.parse_args()
if a.dev_only==bool(a.frozen_config):
    ap.error('Choose --dev_only for parameter validation, or --frozen_config for a test replay')
if a.eval_root and (a.dev_only or not a.evaluation_manifest or not Path(a.evaluation_manifest).is_file()):
    ap.error('External evaluation requires a frozen configuration and an existing evaluation manifest')
frozen=load_frozen_config(a.frozen_config) if a.frozen_config else None
if frozen:
    for flag in ['--grid','--beam','--lex','--no_unk','--comp_w','--ctc_cw','--completed_bank','--bank_size','--end_window','--end_threshold']:
        if any(x==flag or x.startswith(flag+'=') for x in sys.argv[1:]):
            ap.error(f'{flag} is owned by the frozen configuration; choose settings on development data')
    saved=frozen['protocol']
    a.grid=frozen['best'];a.beam=saved['beam'];a.lex=saved['lex'];a.no_unk=saved['no_unk']
    a.comp_w=','.join(map(str,saved['attention_weights']));a.ctc_cw=','.join(map(str,saved['ctc_weights']))
    a.completed_bank=saved.get('completed_bank',False);a.bank_size=saved.get('bank_size',32)
    a.end_window=saved.get('end_window',3);a.end_threshold=saved.get('end_threshold',-10.)
if a.completed_bank and not a.target_full:ap.error('Completed-bank pipeline needs --target_full')
if a.completed_bank and any(parse_parameters(v)[0]!=0 for v in a.grid.split(',')):ap.error('Proposal bank does not apply CTC prefix scores')
target_full=load_frozen_config(a.target_full) if a.target_full else None
TW=(*parse_parameters(target_full['best']),target_full['protocol']['ctc_weights']) if target_full else None
checkpoint=Path(a.checkpoint or f'{a.run}/best.pt')
if not checkpoint.is_file() or not Path(a.lm).is_file():
    ap.error('The acoustic checkpoint and explicit LM file must both exist')
out_dir=Path(a.out_dir or a.run);out_dir.mkdir(parents=True,exist_ok=True)
if any((out_dir/f'{stem}_{a.tag}.json').exists() for stem in ['result','hyps','decoding','nbest']):
    ap.error('Output tag already exists; choose a new tag to preserve recorded evidence')
args = json.load(open(f"{a.run}/result.json"))["args"]
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(0)
T.ROOT = a.root or args.get("root", T.ROOT)
train_all,test=load_replay_splits(T,a.eval_root,a.dev_only)
if args.get("dev_split"):
    train, devset = train_all, T.load_split(args["dev_split"])
else:
    spks = sorted({it["spk"] for it in train_all}); dev_spk = set(spks[-3:])
    train = [it for it in train_all if it["spk"] not in dev_spk]; devset = [it for it in train_all if it["spk"] in dev_spk]
if args.get("dev_max") and len(devset) > args["dev_max"]:
    random.Random(0).shuffle(devset); devset = devset[: args["dev_max"]]
voc = T.Vocab(train, factor=args["factor_rhyme"])
if a.limit:
    if a.limit<0:ap.error('--limit must be nonnegative')
    devset,test=devset[:a.limit],test[:a.limit]
train_sents = {" ".join(it["words"]) for it in train_all}
for it in devset + test:
    it["audio"] = T.read_wav(it["wav"]); it["dur"] = len(it["audio"]) / 16000; it["sent"] = " ".join(it["words"])
fe = (T.WavFrontend() if args["ssl"] else T.Frontend()).to(dev)
model = T.Model(voc, enc_layers=args["enc_layers"], ssl=args["ssl"]).to(dev)
model.load_state_dict(torch.load(checkpoint,map_location=dev)); model.eval()
ck = torch.load(a.lm); itos = ck["itos"]; stoi = {w: i for i, w in enumerate(itos)}
lm = LM(len(itos)).to(dev); lm.load_state_dict(ck["sd"]); lm.eval()

# candidate syllable inventory
if a.lex == "train":
    trips = {t for it in train for t in voc.enc(it["trip"])}
else:
    trips = set()
    for w in itos[4:]:
        p = tok.parse(w)
        if p and tok.roundtrip_ok(w):
            ids = voc.enc([p])[0]
            if T.UNK not in ids: trips.add(ids)
    trips |= {t for it in train for t in voc.enc(it["trip"]) if T.UNK not in t}
trips.discard((T.UNK,)*voc.nc)
has_unk = (not a.no_unk) and any(T.UNK in t for it in train for t in voc.enc(it["trip"]))
trips = sorted(trips) + ([(T.UNK,) * voc.nc] if has_unk else []) + [(T.EOS,) * voc.nc]
allowed = torch.tensor(trips, device=dev); A = len(trips); EOSI = A - 1
UNKI = trips.index((T.UNK,) * voc.nc) if has_unk else -1
def surf(t):
    return tok.surface(*voc.dec(list(t)))
def _surf(t):
    return None if T.UNK in t else surf(t)
lmid = torch.tensor([stoi.get(_surf(t), 3) if _surf(t) else 3 for t in trips[:-1]] + [2], device=dev)
unp_ids = torch.tensor([i for i, w in enumerate(itos) if i >= 4 and not tok.roundtrip_ok(w)] or [3], device=dev)   # LM tokens that cannot be spelled by the decoder
print("candidates", A, "lm-known", int((lmid[:-1] != 3).sum()), "unk-candidate", has_unk, "unp-lm-tokens", len(unp_ids), flush=True)

CW = [float(x) for x in a.comp_w.split(",")]
CCW = [float(x) for x in a.ctc_cw.split(",")]
if len(CW)!=voc.nc or len(CCW)!=voc.nc or any(not math.isfinite(x) or x<0 for x in CW+CCW):
    ap.error('Component weights must be finite, nonnegative and match the number of heads')
if a.beam<1:ap.error('Beam width must be positive')
source_dir=Path(__file__).resolve().parent
protocol=dict(version=3,evaluation_scope=('development_only' if a.dev_only else ('exploratory_reused_external_test' if a.eval_root else 'exploratory_reused_in_domain_test')),independent_confirmation=False,
    checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),lm_sha256=file_sha256(a.lm),
    inventory_sha256=hashlib.sha256(json.dumps([trips,lmid.cpu().tolist()],separators=(',',':')).encode()).hexdigest(),
    attention_weights=CW,ctc_weights=CCW,beam=a.beam,lex=a.lex,no_unk=a.no_unk,
    bounds='min(valid encoder frames + 1 EOS step, model positional capacity), separately per utterance',
    reference_dependent_search=False,unknown_policy_selection='historically adaptive; frozen replay does not undo exposure',
    search_family='separate_completed_bank_proposal' if a.completed_bank else 'instrumented_legacy_beam',max_batch_seconds=a.max_sec,subset_limit=a.limit,
    dev_utterances=len(devset),test_utterances=len(test),seed=args.get('seed'),data_root=T.ROOT,
    eval_data_root=a.eval_root or T.ROOT,
    external_evaluation_manifest_sha256=file_sha256(a.evaluation_manifest) if a.eval_root else None,
    negative_infinity_encoding='-inf',software={'torch':torch.__version__,'python':sys.version.split()[0]},
    code_sha256={name:file_sha256(source_dir/name) for name in ['decode_lm.py','decoding_protocol.py','nbest_rescore.py','replay_manifest.py','train_vivos.py','tok.py','lm_model.py','ctc_prefix.py','bank_search.py','rescore_saved_nbest.py']})
protocol.update(completed_bank=a.completed_bank,bank_size=a.bank_size,end_window=a.end_window,
    end_threshold=a.end_threshold,pipeline_target_full_sha256=file_sha256(a.target_full) if target_full else None,
    pipeline_fallback='unconstrained greedy only for empty finite completed target-score pool',
    proposal_selection='development reranked pipeline WER; no test-based selection')
if target_full:
    for key in ['checkpoint_sha256','lm_sha256','inventory_sha256','attention_weights','ctc_weights']:
        if target_full['protocol'][key]!=protocol[key]:ap.error('Target model/head/inventory resource mismatch: '+key)
if frozen:
    verify_resources(frozen['protocol'],protocol)
    for key in ['completed_bank','bank_size','end_window','end_threshold','pipeline_target_full_sha256','beam','lex','no_unk','max_batch_seconds','data_root','seed']:
        if frozen['protocol'].get(key)!=protocol.get(key):ap.error('Frozen bank protocol mismatch: '+key)

print('PROTOCOL',json.dumps(protocol),flush=True)

@torch.no_grad()
def beam(feats, n, max_len, alpha, beta, K, lam=0.0, return_info=False,return_nbest=False):
    if a.completed_bank:
        out,info,pool=bank_beam(globals(),feats,n,max_len,alpha,beta,K,
            bank_size=a.bank_size,end_window=a.end_window,end_threshold=a.end_threshold,target=TW)
        return (out,info,pool) if return_nbest else ((out,info) if return_info else out)
    mem, mm = model.encode(feats, n)
    B = feats.size(0); NC = voc.nc
    position_buffer=getattr(getattr(model,'pe',None),'pe',None)
    if position_buffer is not None:
        capacity=int(position_buffer.shape[-2])
        max_len=capacity if max_len is None else min(max_len,capacity)
    tracker = DecodeTracker(mm, max_len)
    acoustic_mem, acoustic_mask = mem, mm
    ctcs = []
    if lam > 0:                                                                      # factorized CTC prefix scorers, one per component
        lens = (~mm).sum(1).repeat_interleave(K, 0)
        for j, head in enumerate(model.cheads):
            if CCW[j] == 0: ctcs.append(None); continue
            ctcs.append(ComponentCTC(F.log_softmax(head(mem).float(), -1).repeat_interleave(K, 0), lens, blank=T.PAD))
    mem = mem.repeat_interleave(K, 0); mm = mm.repeat_interleave(K, 0)
    seq = torch.full((B * K, 1, NC), T.BOS, dtype=torch.long, device=dev)
    lmin = torch.full((B * K, 1), 1, dtype=torch.long, device=dev)
    score = torch.zeros(B, K, device=dev); score[:, 1:] = -1e9
    fin = torch.zeros(B, K, dtype=torch.bool, device=dev)
    ended_evictions=torch.zeros(B,dtype=torch.long,device=dev)
    attention_sum = torch.zeros(B,K,dtype=torch.float64,device=dev)
    lm_sum = torch.zeros_like(attention_sum)
    length_sum = torch.zeros(B,K,dtype=torch.long,device=dev)
    for step in range(int(tracker.limits.max())):
        lps = [F.log_softmax(x[:, -1].float(), -1) for x in model.decode(seq, mem, mm)]
        am = sum(CW[j] * lp[:, allowed[:, j]] for j, lp in enumerate(lps))          # BK,A
        lml = torch.zeros_like(am)
        if alpha > 0 or return_nbest:
            lfull = F.log_softmax(lm(lmin[:, -CTX:])[:, -1].float(), -1)
            lml = lfull[:, lmid]                                                         # BK,A
            if UNKI >= 0:
                lml[:, UNKI] = torch.logsumexp(lfull[:, unp_ids], -1)                    # <unk> = any non-spellable token
        add = (1-lam)*am + alpha*lml
        if lam > 0:
            cadd = torch.zeros(B * K, A, dtype=torch.float64, device=dev)
            for j, st in enumerate(ctcs):
                if st is None: continue
                psi, end = st.scores()
                cj = psi[:, allowed[:, j]] - st.psi[:, None]
                cj[:, EOSI] = end - st.psi
                cadd = cadd + CCW[j] * cj
            add = add + lam*cadd.float()
        add = add + beta * torch.cat([torch.ones(A - 1, device=dev), torch.zeros(1, device=dev)])
        add = add.view(B, K, A)
        inactive = tracker.inactive(fin,step)
        add = torch.where(inactive[..., None], torch.full_like(add, -1e9), add)
        add[..., EOSI] = torch.where(inactive, torch.zeros_like(add[..., EOSI]), add[..., EOSI])
        tot = (score[..., None] + add).view(B, K * A)
        score, ix = tot.topk(K, -1)
        par, tk = ix // A, ix % A
        kept_parents=torch.zeros_like(fin);kept_parents.scatter_(1,par,True)
        live_rows=(tracker.stop_steps==0)&(step<tracker.limits)
        ended_evictions+=(fin&~kept_parents&live_rows[:,None]).sum(1)
        base = (torch.arange(B, device=dev) * K)[:, None]
        sel = (base + par).view(-1)
        active = ~inactive.gather(1,par)
        batch = torch.arange(B,device=dev)[:,None]
        attention_sum = attention_sum.gather(1,par)+torch.where(active,am.view(B,K,A)[batch,par,tk].double(),0.)
        lm_sum = lm_sum.gather(1,par)+torch.where(active,lml.view(B,K,A)[batch,par,tk].double(),0.)
        length_sum = length_sum.gather(1,par)+(active & (tk!=EOSI)).long()
        seq = torch.cat([seq[sel], allowed[tk.view(-1)][:, None]], 1)
        lmin = torch.cat([lmin[sel], lmid[tk.view(-1)][:, None]], 1)
        for j, st in enumerate(ctcs):
            if st is not None: st.select(sel, allowed[tk.view(-1)][:, j])
        fin = fin.gather(1,par) | ((tk==EOSI)&active)
        tracker.update(fin,step)
        if tracker.inactive(fin,step+1).all():break
    best = score.argmax(-1)
    sequences = seq.view(B,K,-1,NC)[:,:,1:]
    candidates = [[] for _ in range(B)]
    if return_nbest:
        for j in range(K):
            output,metadata=tracker.finalize(sequences,fin,torch.full((B,),j,device=dev))
            for b,(tokens,info) in enumerate(zip(output,metadata)):
                labels=[x for x in tokens if T.EOS not in x]
                candidates[b].append(dict(component_ids=labels,
                    attention_logprob=float(attention_sum[b,j]),lm_logprob=float(lm_sum[b,j]),
                    length_bonus_count=int(length_sum[b,j]),ended=info['selected_eos'],
                    search_score=float(score[b,j]),beam_index=j))
        # Complete CTC values are exported AFTER search, including when lambda=0.
        # The saved LM proposal list can then be rescored on CPU without a model.
        log_probs=[F.log_softmax(head(acoustic_mem).float(),-1) for head in model.cheads]
        for b in range(B):
            ranked=rescore_component_nbest(candidates[b],[x[b] for x in log_probs],
                int((~acoustic_mask[b]).sum()),lam,alpha,beta,CCW,blank=T.PAD)
            by_index={x['beam_index']:x for x in ranked}
            candidates[b]=[by_index.get(x['beam_index'],dict(x,ctc_logprobs=None,rescored_score=None)) for x in candidates[b]]
    output,info=tracker.finalize(sequences,fin,best)
    for b,row in enumerate(info):
        row['legacy_ended_records_evicted']=int(ended_evictions[b])
        row['legacy_ended_evicted']=bool(ended_evictions[b])
    if return_nbest:return output,info,candidates
    return (output,info) if return_info else output

def run(items, alpha, beta, K, want=False, lam=0.0, decoding_log=None, nbest_log=None,pipeline_store=None):
    hyps = {}
    last_report=time.monotonic()
    for idx in T.batches(items, a.max_sec, False, None):
        wav, wl = T.collate_audio(items,idx,dev)
        with torch.autocast(dev.type,dtype=torch.bfloat16,enabled=dev.type=='cuda'):
            feats, n = fe(wav, wl)
            need_nbest=nbest_log is not None or pipeline_store is not None or a.completed_bank
            result = beam(feats,n,None,alpha,beta,K,lam,return_info=True,return_nbest=need_nbest)
            if need_nbest:out,info,proposals=result
            else:out,info=result
        for b,(k,s,record) in enumerate(zip(idx,out,info)):
            hyps[k]=T.ids_to_words(s,voc)
            if decoding_log is not None:decoding_log[items[k]['uid']]=record
            if need_nbest:
                for candidate in proposals[b]:candidate['hyp']=' '.join(T.ids_to_words(candidate['component_ids'],voc)[0])
                if nbest_log is not None:nbest_log[items[k]['uid']]=proposals[b]
            if pipeline_store is not None:
                chosen,counts=_rank(json_safe(proposals[b]),TW)
                if chosen is not None:
                    labels=chosen[1]['component_ids']
                    record['pipeline_choice']=dict(source='completed_pool',candidate_index=chosen[0],
                        beam_index=chosen[1].get('beam_index'),target_score=chosen[1]['rescored_score'],counts=counts)
                else:
                    if 'greedy_fallback_component_ids' not in record:
                        fallback,finfo=T.greedy(model,feats[b:b+1],n[b:b+1],None,return_info=True)
                        record['greedy_fallback_component_ids']=fallback[0]
                        record['greedy_fallback_termination']=finfo[0]
                    labels=record['greedy_fallback_component_ids']
                    record['pipeline_choice']=dict(source='empty_pool_unconstrained_greedy',counts=counts)
                pipeline_store[k]=T.ids_to_words(labels,voc)

        if time.monotonic()-last_report>=30:
            print(f'DECODE {len(hyps)}/{len(items)} utterances',flush=True)
            last_report=time.monotonic()
    return T.score(items, hyps, want)


def save_pool(path,records):
    with gzip.open(path,'xt',encoding='utf-8',compresslevel=6) as stream:
        json.dump(json_safe(records),stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        stream.write('\n')


def execute_split(items,parameters,tag):
    lam,al,be=parse_parameters(parameters)
    logs,pools,pipeline={},{},{}
    started=time.time()
    proposal,rows,per=run(items,al,be,a.beam,True,lam=lam,decoding_log=logs,nbest_log=pools,
                          pipeline_store=pipeline if target_full else None)
    pipe,pipe_rows,pipe_per=T.score(items,pipeline,True) if target_full else (proposal,rows,per)
    dump_record(out_dir/f'hyps_{tag}.json',per)
    dump_record(out_dir/f'pipeline_hyps_{tag}.json',pipe_per)
    dump_record(out_dir/f'decoding_{tag}.json',dict(protocol=protocol,utterances=logs))
    save_pool(out_dir/f'nbest_{tag}.json.gz',dict(protocol=protocol,utterances=pools,
        target_full_frozen_sha256=protocol['pipeline_target_full_sha256']))
    record=dict(best=parameters,proposal_metrics=proposal,pipeline_metrics=pipe,protocol=protocol,
        artifact_sha256={f.name:file_sha256(f) for f in [out_dir/f'hyps_{tag}.json',out_dir/f'pipeline_hyps_{tag}.json',out_dir/f'decoding_{tag}.json',out_dir/f'nbest_{tag}.json.gz']},
        termination=dict(utterances=len(logs),selected_cap=sum(not r['selected_eos'] for r in logs.values()),
            any_beam_cap=sum(r['search_cap_hit'] for r in logs.values()),
            end_detected=sum(r.get('end_detected',False) for r in logs.values()),
            pipeline_fallback=sum(r.get('pipeline_choice',{}).get('source')=='empty_pool_unconstrained_greedy' for r in logs.values()),
            pipeline_selected_cap=sum(r.get('pipeline_choice',{}).get('source')=='empty_pool_unconstrained_greedy' and not r.get('greedy_fallback_termination',{}).get('selected_eos',False) for r in logs.values()),
            legacy_ended_evicted_utterances=sum(r.get('legacy_ended_evicted',False) for r in logs.values()),
            legacy_ended_records_evicted=sum(r.get('legacy_ended_records_evicted',0) for r in logs.values())),
        decode_seconds=time.time()-started)
    dump_record(out_dir/f'point_{tag}.json',record)
    print('PIPELINE',parameters,json.dumps(pipe),'PROPOSAL',json.dumps(proposal),'SECONDS',record['decode_seconds'],flush=True)
    return record


if a.dev_only:
    res={};proposal_grid={};points={}
    for index,g in enumerate(a.grid.split(',')):
        parameters=':'.join(format(x,'.17g') for x in parse_parameters(g))
        point_tag=f'{a.tag}_{index:02d}'
        point_path=out_dir/f'point_{point_tag}.json'
        paths=[out_dir/f'{stem}_{point_tag}{extension}' for stem,extension in
            [('point','.json'),('hyps','.json'),('pipeline_hyps','.json'),('decoding','.json'),('nbest','.json.gz')]]
        if all(path.is_file() for path in paths):
            rec=json.loads(point_path.read_text())
            if rec['protocol']!=protocol or parse_parameters(rec['best'])!=parse_parameters(parameters):
                raise ValueError('Existing development point has different source/settings')
            for name,digest in rec['artifact_sha256'].items():
                if file_sha256(out_dir/name)!=digest:raise ValueError('Existing development artifact hash mismatch')
            print('RESUMED',parameters,flush=True)
        else:
            present=[path for path in paths if path.exists()]
            if present:
                archive=out_dir/('partial_'+point_tag+'_'+str(time.time_ns()));archive.mkdir()
                for path in present:shutil.move(str(path),archive/path.name)
            rec=execute_split(devset,parameters,point_tag)

        res[parameters]=rec['pipeline_metrics'];proposal_grid[parameters]=rec['proposal_metrics']
        points[parameters]=dict(path=f'point_{a.tag}_{index:02d}.json',sha256=file_sha256(out_dir/f'point_{a.tag}_{index:02d}.json'))
    bg=min(res,key=lambda k:(res[k]['WER'],res[k]['CER'],abs(parse_parameters(k)[2]),parse_parameters(k)[1],parse_parameters(k)[2]))
    dump_record(out_dir/f'result_{a.tag}.json',dict(best=bg,selection_scope='development',
        selection_rule='minimum_development_pipeline_WER_then_CER_absbonus_alpha_bonus',
        protocol=protocol,dev_grid=res,proposal_dev_grid=proposal_grid,point_records=points))
    sys.exit(0)
record=execute_split(test,frozen['best'],a.tag)
record.update(best=frozen['best'],test=record['pipeline_metrics'],proposal_test=record['proposal_metrics'],
              dev_grid=frozen['dev_grid'],frozen_config=str(Path(a.frozen_config)),
              frozen_config_sha256=file_sha256(a.frozen_config),selection_scope='frozen_development_test')
dump_record(out_dir/f'result_{a.tag}.json',record)
