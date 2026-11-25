"""A separate completed bank; no references or CTC terms guide proposals.

End detection implements the score-gap rule of Hori et al. (2017), Eq.18,
with fixed M=3/threshold=-10 as in ESPnet v.202402 e2e_asr_common.py.
Per-length maxima retain discarded completions; the proposal bank is bounded.
"""
import math


class CompletedBank:
    def __init__(self, size):
        if isinstance(size,bool) or not isinstance(size,int) or size<1:
            raise ValueError('Completed bank size must be positive')
        self.size=size
        self._rows={}
        self.by_length={}
        self.observed=0
        self.bank_pruned=0
        self._serial=0

    def observe(self, records, length):
        for record in records:
            if record.get('ended') is not True or record['length_bonus_count']!=length:
                raise ValueError('Only actual EOS records of the stated nonterminal length are allowed')
            score=float(record['search_score'])
            if not math.isfinite(score):continue
            self.observed+=1
            self.by_length[length]=max(self.by_length.get(length,-math.inf),score)
            key=tuple(tuple(ids) for ids in record['component_ids'])
            previous=self._rows.get(key)
            if previous is None or score>previous[0]['search_score']:
                self._rows[key]=(dict(record), self._serial)
                self._serial+=1
        ranked=sorted(self._rows.items(),key=lambda kv:(-kv[1][0]['search_score'],kv[1][1]))
        self.bank_pruned+=max(0,len(ranked)-self.size)
        self._rows=dict(ranked[:self.size])

    def records(self):
        return [pair[0] for pair in sorted(self._rows.values(),key=lambda p:(-p[0]['search_score'],p[1]))]

    def best(self):
        rows=self.records()
        return rows[0] if rows else None

    def end_detected(self, length, M=3, threshold=-10.):
        best=self.best()
        if best is None:return False
        return all((length-m in self.by_length and
                    self.by_length[length-m]-best['search_score']<threshold)
                   for m in range(M))

import torch
import torch.nn.functional as F


@torch.no_grad()
def bank_beam(g, feats, n, max_len, alpha, beta, K, *, bank_size=32,
              end_window=3, end_threshold=-10., target):
    """Decode without CTC prefix guidance; export exact completed target scores.

    g is the existing decoder context (model, lm, T, voc, allowed, lmid, CW,
    EOSI, UNKI, unp_ids, CTX). Only those named fields are consumed. The target
    tuple is (lambda, alpha, beta, component weights), fixed before this call.
    The separate live beam never contains EOS. References are not inputs.
    """
    from nbest_rescore import rescore_component_nbest
    from decoding_protocol import DecodeTracker
    model,lm,T=g['model'],g['lm'],g['T']
    allowed,lmid=g['allowed'],g['lmid']
    CW=g['CW'];EOSI=g['EOSI'];NC=g['voc'].nc
    if K<1 or bank_size<1 or end_window<1 or not math.isfinite(end_threshold) or end_threshold>=0:
        raise ValueError('Positive beam/bank/window and negative finite end threshold required')
    memory,mask=model.encode(feats,n)
    device=memory.device;B=len(feats);A=len(allowed)
    pe=getattr(getattr(model,'pe',None),'pe',None)
    guard=int(pe.shape[-2]) if pe is not None else 4000
    if max_len is not None:guard=min(guard,max_len)
    limits=DecodeTracker(mask,guard)
    frames=limits.encoder_frames.detach().cpu().tolist()
    bounds=limits.limits.detach().cpu().tolist()
    banks=[CompletedBank(bank_size) for _ in range(B)]
    stopped=[False]*B;steps=[0]*B;end_detected=[False]*B;cap_hit=[False]*B
    completion_expansions=[0]*B
    mem=memory.repeat_interleave(K,0);mm=mask.repeat_interleave(K,0)
    seq=torch.full((B*K,1,NC),T.BOS,dtype=torch.long,device=device)
    lmin=torch.full((B*K,1),1,dtype=torch.long,device=device)
    score=torch.full((B,K),-math.inf,device=device);score[:,0]=0.
    attention_sum=torch.zeros(B,K,dtype=torch.float64,device=device)
    lm_sum=torch.zeros_like(attention_sum)
    lengths=torch.zeros(B,K,dtype=torch.long,device=device)
    base=(torch.arange(B,device=device)*K)[:,None]
    for step in range(max(bounds)):
        lps=[F.log_softmax(x[:,-1].float(),-1) for x in model.decode(seq,mem,mm)]
        am=sum(CW[j]*lp[:,allowed[:,j]] for j,lp in enumerate(lps))
        lf=F.log_softmax(lm(lmin[:,-g['CTX']:])[:,-1].float(),-1)
        lml=lf[:,lmid]
        if g['UNKI']>=0:lml[:,g['UNKI']]=torch.logsumexp(lf[:,g['unp_ids']],-1)
        am3=am.view(B,K,A);lm3=lml.view(B,K,A)
        # EOS gets all attention heads and one LM term, without a bonus.
        eos_score=score+am3[:,:,EOSI]+alpha*lm3[:,:,EOSI]
        eos_a=attention_sum+am3[:,:,EOSI].double()
        eos_l=lm_sum+lm3[:,:,EOSI].double()
        values=eos_score.detach().cpu().tolist()
        avalues=eos_a.detach().cpu().tolist();lvalues=eos_l.detach().cpu().tolist()
        seq_cpu=None
        for b in range(B):
            if stopped[b]:continue
            rows=[]
            # Materialize only completions that could enter the bank, plus the
            # best current length (needed by end detection even when discarded).
            existing=banks[b].records()
            cutoff=existing[-1]['search_score'] if len(existing)==bank_size else -math.inf
            valid=[j for j,s in enumerate(values[b]) if math.isfinite(s)]
            completion_expansions[b]+=len(valid)
            keep=[j for j in valid if values[b][j]>=cutoff]
            if valid:
                current=max(valid,key=lambda j:values[b][j])
                if current not in keep:keep.append(current)
            if keep and seq_cpu is None:seq_cpu=seq.view(B,K,-1,NC)[:,:,1:].detach().cpu().tolist()
            for j in keep:
                rows.append(dict(component_ids=seq_cpu[b][j],
                    attention_logprob=avalues[b][j],lm_logprob=lvalues[b][j],
                    length_bonus_count=step,ended=True,search_score=values[b][j],
                    beam_index=step*K+j))
            banks[b].observe(rows,step)
            end_detected[b]=banks[b].end_detected(step,end_window,end_threshold)
            cap_hit[b]=step+1>=bounds[b]
            if end_detected[b] or cap_hit[b]:stopped[b]=True;steps[b]=step+1
        add=am3+alpha*lm3+beta
        add[:,:,EOSI]=-math.inf  # EOS records are stored only in the completed bank.
        inactive=torch.tensor(stopped,dtype=torch.bool,device=device)
        total=score[:,:,None]+add
        total=total.masked_fill(inactive[:,None,None],-math.inf)
        score,index=total.view(B,K*A).topk(K,-1)
        parent,token=index//A,index%A
        selected=(base+parent).view(-1)
        batch=torch.arange(B,device=device)[:,None]
        active=~inactive[:,None]
        attention_sum=attention_sum.gather(1,parent)+torch.where(active,am3[batch,parent,token].double(),0.)
        lm_sum=lm_sum.gather(1,parent)+torch.where(active,lm3[batch,parent,token].double(),0.)
        lengths=lengths.gather(1,parent)+active.long()
        seq=torch.cat([seq[selected],allowed[token.view(-1)][:,None]],1)
        lmin=torch.cat([lmin[selected],lmid[token.view(-1)][:,None]],1)
        if all(stopped):break
    log_probs=[F.log_softmax(head(memory).float(),-1) for head in model.cheads]
    output=[];infos=[];pools=[]
    for b,bank in enumerate(banks):
        candidates=bank.records()
        ranked=rescore_component_nbest(candidates,[lp[b] for lp in log_probs],frames[b],
                                        target[0],target[1],target[2],target[3],blank=T.PAD)
        by_index={row['beam_index']:row for row in ranked}
        candidates=[by_index[row['beam_index']] for row in candidates]
        best=bank.best()
        if best is None:
            # A finite-softmax model normally has an EOS candidate at step zero.
            # This fallback is explicit for failure-domain completeness.
            chosen=[]
        else:chosen=best['component_ids']+[[T.EOS]*NC]
        output.append(chosen);pools.append(candidates)
        info=dict(encoder_frames=frames[b],effective_limit=bounds[b],
                  limit_source='encoder_valid_frames_plus_eos',fixed_guard=guard,
                  stop_steps=steps[b],selected_output_steps=len(chosen),
                  termination='eos' if best is not None else 'cap',selected_eos=best is not None,
                  search_cap_hit=cap_hit[b],end_detected=end_detected[b],
                  completed_beams=len(candidates),total_beams=bank_size,
                  completed_bank_size=len(candidates),active_beam_width=K,
                  materialized_completions=bank.observed,bank_pruned_events=bank.bank_pruned,
                  completed_expansions=completion_expansions[b],
                  bank_size_limit=bank_size,end_window=end_window,end_threshold=end_threshold)
        finite=[row for row in candidates if math.isfinite(row['rescored_score'])]
        if not finite:
            greedy,ginfo=T.greedy(model,feats[b:b+1],n[b:b+1],None,return_info=True)
            info['greedy_fallback_component_ids']=greedy[0]
            info['greedy_fallback_termination']=ginfo[0]
            info['pipeline_fallback']='empty_finite_completed_pool'
        else:info['pipeline_fallback']=None
        infos.append(info)
    return output,infos,pools
