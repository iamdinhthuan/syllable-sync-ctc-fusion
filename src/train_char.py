"""Character-level baseline with the SAME encoder/training recipe as train_vivos.py (XLS-R + hybrid CTC/attention), optional char-LM shallow fusion."""
import argparse, collections, json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from rapidfuzz.distance import Levenshtein as Lev
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_vivos as T
from train_vivos import PAD, BOS, EOS, UNK
from lm_model import LM
from decoding_protocol import DecodeTracker


class CharVocab:
    def __init__(self, itos):
        self.itos = itos; self.vocs = [itos]; self.xs = [{c: i for i, c in enumerate(itos)}]; self.nc = 1
    def enc(self, chars):
        return [(self.xs[0].get(c, UNK),) for c in chars]


def ids_to_str(ids, voc):
    out = []
    for t in ids:
        k = t[0] if isinstance(t, (list, tuple)) else t
        if k == EOS: break
        if k in (PAD, BOS): continue
        out.append("⁇" if k == UNK else voc.itos[k])
    return "".join(out)


def to_hyp(s):
    w = s.split()
    return w, [("<unk>",) * 3] * len(w)


def prep(items):
    for it in items:
        it["trip"] = list(" ".join(it["words"]))            # characters of the canonical reference
        it["audio"] = T.read_wav(it["wav"]); it["dur"] = len(it["audio"]) / 16000; it["sent"] = " ".join(it["words"])


def score_char(items, hyps, want=False):
    tw = ew = tc = ec = 0; rows = []; per = {}
    for k, it in enumerate(items):
        hw = hyps[k]; rw = it["words"]
        a = Lev.distance(rw, hw); r1 = " ".join(rw); h1 = " ".join(hw); b = Lev.distance(r1, h1)
        tw += len(rw); ew += a; tc += len(r1); ec += b
        rows.append((it["spk"], it["sent"], len(rw), a, b, len(r1), 0, 0, 0))
        if want: per[it["uid"]] = dict(ref=r1, hyp=h1)
    return dict(WER=100 * ew / tw, CER=100 * ec / tc), rows, per


def evaluate(model, fe, items, voc, dev, want=False, beam_fn=None, max_sec=200, decoding_log=None):
    model.eval(); hyps = {}
    for idx in T.batches(items, max_sec, False, None):
        wav, wl = T.collate_audio(items, idx, dev)
        with torch.autocast(wav.device.type, dtype=torch.bfloat16, enabled=wav.is_cuda):
            feats, n = fe(wav, wl)
            kwargs = {"return_info": True} if decoding_log is not None else {}
            result = beam_fn(feats, n, None, **kwargs) if beam_fn else T.greedy(model, feats, n, None, **kwargs)
        if decoding_log is not None:
            out, records = result
            decoding_log.extend(dict(uid=items[k]["uid"], **record) for k, record in zip(idx, records))
        else:
            out = result
        for k, s in zip(idx, out): hyps[k] = ids_to_str(s, voc).split()
    return score_char(items, hyps, want)


@torch.no_grad()
def beam_char(model, lm, ctx, voc, dev, feats, n, max_len, alpha, beta, K, return_info=False):
    """Character beam search with audio-only limits and optional selected-beam termination records."""
    mem, mm = model.encode(feats, n)
    position_buffer = getattr(getattr(model, "pe", None), "pe", None)
    if position_buffer is not None and (max_len is None or isinstance(max_len, int)):
        capacity = int(position_buffer.shape[-2])
        max_len = capacity if max_len is None else min(max_len, capacity)
    tracker = DecodeTracker(mm, max_len)
    B = feats.size(0); V = len(voc.itos)
    mem = mem.repeat_interleave(K, 0); mm = mm.repeat_interleave(K, 0)
    seq = torch.full((B * K, 1, 1), BOS, dtype=torch.long, device=dev)
    lmin = torch.full((B * K, 1), BOS, dtype=torch.long, device=dev)
    score = torch.zeros(B, K, device=dev); score[:, 1:] = -1e9
    fin = torch.zeros(B, K, dtype=torch.bool, device=dev)
    bad = torch.zeros(V, dtype=torch.bool, device=dev); bad[[PAD, BOS, UNK]] = True
    bonus = torch.full((V,), float(beta), device=dev); bonus[EOS] = 0.0
    for step in range(int(tracker.limits.max())):
        inactive = tracker.inactive(fin, step)
        add = F.log_softmax(model.decode(seq, mem, mm)[0][:, -1].float(), -1)
        if alpha > 0:
            add = add + alpha * F.log_softmax(lm(lmin[:, -ctx:])[:, -1].float(), -1)
        add = (add + bonus).masked_fill(bad, -1e9).view(B, K, V)
        add = add.masked_fill(inactive[..., None], -float("inf"))
        add[..., PAD] = torch.where(inactive, torch.zeros_like(add[..., PAD]), add[..., PAD])
        old_score = score
        score, ix = (score[..., None] + add).view(B, K * V).topk(K, -1)
        par, tk = ix // V, ix % V
        paused = inactive.all(-1)
        par = torch.where(paused[:, None], torch.arange(K, device=dev)[None], par)
        tk = torch.where(paused[:, None], torch.full_like(tk, PAD), tk)
        score = torch.where(paused[:, None], old_score, score)
        carried = inactive.gather(1, par)
        sel = ((torch.arange(B, device=dev) * K)[:, None] + par).view(-1)
        seq = torch.cat([seq[sel], tk.view(-1, 1, 1)], 1)
        lmin = torch.cat([lmin[sel], tk.view(-1, 1)], 1)
        fin = fin.gather(1, par) | ((tk == EOS) & ~carried)
        tracker.update(fin, step)
        if (fin | (step + 1 >= tracker.limits[:, None])).all(): break
    best = score.argmax(-1)
    out, records = tracker.finalize(seq.view(B, K, -1, 1)[:, :, 1:], fin, best, eos_id=EOS)
    for b, (sequence, record) in enumerate(zip(out, records)):
        text_ids = [x[0] for x in sequence if x[0] != EOS]
        record.update(selected_score=float(score[b, best[b]]), lm_history=[BOS] + text_ids,
                      partial_word=bool(not record["selected_eos"] and text_ids and voc.itos[text_ids[-1]] != " "))
    return (out, records) if return_info else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--chars", required=True)
    ap.add_argument("--root", default=T.ROOT); ap.add_argument("--dev_split", default=""); ap.add_argument("--dev_max", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--ssl", default="facebook/wav2vec2-xls-r-300m"); ap.add_argument("--ssl_lr", type=float, default=3e-5)
    ap.add_argument("--max_sec", type=float, default=80.0); ap.add_argument("--ctc_w", type=float, default=0.3)
    ap.add_argument("--lm", default=""); ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--grid", default=",".join(f"{al}:{be}" for al in (0.3, 0.5, 0.7, 0.9) for be in (0.5, 1, 2)))
    ap.add_argument("--decode_only", action="store_true", help="skip training; load best.pt in --out")
    ap.add_argument("--keep_ckpt", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda"); T.ROOT = a.root
    train_all, test = T.load_split("train"), T.load_split("test")
    if a.dev_split: train, devset = train_all, T.load_split(a.dev_split)
    else:
        spks = sorted({it["spk"] for it in train_all}); ds_ = set(spks[-3:])
        train = [it for it in train_all if it["spk"] not in ds_]; devset = [it for it in train_all if it["spk"] in ds_]
    if a.dev_max and len(devset) > a.dev_max:
        random.Random(0).shuffle(devset); devset = devset[: a.dev_max]
    if a.limit:
        random.Random(0).shuffle(train); train = train[: a.limit]; devset = devset[: max(a.limit // 5, 10)]; test = test[: max(a.limit // 5, 10)]
    voc = CharVocab(json.load(open(a.chars, encoding="utf-8")))
    if a.train_frac < 1:
        random.Random(0).shuffle(train); train = train[: int(len(train) * a.train_frac)]; print('train_frac', a.train_frac, 'utts', len(train), flush=True)
    train_sents = {" ".join(it["words"]) for it in train_all}
    for part in (train, devset, test): prep(part)
    print(f"train {len(train)} dev {len(devset)} test {len(test)} | chars={len(voc.itos)}", flush=True)
    fe = T.WavFrontend().to(dev)
    model = T.Model(voc, ssl=a.ssl).to(dev)
    log = []
    if not a.decode_only:
        ssl_p = [p for n_, p in model.named_parameters() if n_.startswith("ssl.") and p.requires_grad]
        oth_p = [p for n_, p in model.named_parameters() if not n_.startswith("ssl.")]
        opt = torch.optim.AdamW([dict(params=oth_p, lr=a.lr), dict(params=ssl_p, lr=a.ssl_lr)], betas=(0.9, 0.98), weight_decay=0.01)
        total = len(T.batches(train, a.max_sec, False, None)) * a.epochs
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))))
        rng = random.Random(a.seed); best, bad = 1e9, 0
        for ep in range(1, a.epochs + 1):
            model.train(); t0 = time.time(); tl = 0.0; nb = 0
            for idx in T.batches(train, a.max_sec, True, rng):
                wav, wl, tgt = T.collate(train, idx, voc, dev, True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats, n = fe(wav, wl)
                    mem, mm = model.encode(feats, n)
                    lg = model.decode(tgt[:, :-1], mem, mm)[0]
                    loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), tgt[:, 1:, 0].reshape(-1), ignore_index=PAD, label_smoothing=0.1)
                    if a.ctc_w > 0: loss = loss + a.ctc_w * T.ctc_loss(model, mem, mm, tgt)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step(); tl += float(loss); nb += 1
            m, _, _ = evaluate(model, fe, devset, voc, dev)
            log.append(dict(epoch=ep, loss=tl / nb, **{f"dev_{k}": v for k, v in m.items()}))
            print(f"ep {ep:3d} loss {tl/nb:.3f} dev CER {m['CER']:.2f} WER {m['WER']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            if m["CER"] < best - 1e-4: best, bad = m["CER"], 0; torch.save(model.state_dict(), f"{a.out}/best.pt")
            else:
                bad += 1
                if bad >= a.patience: print("early stop", flush=True); break
    model.load_state_dict(torch.load(f"{a.out}/best.pt")); model.eval()
    decoding_log = []
    m, rows, per = evaluate(model, fe, test, voc, dev, want=True, decoding_log=decoding_log)
    ana = T.run_analysis(rows, train_sents, a.seed); ana.pop("WER_per_speaker", None)
    print("TEST", json.dumps(m), flush=True)
    res = dict(args=vars(a), test=m, analysis=ana, log=log, decoding_log=decoding_log)
    json.dump(per, open(f"{a.out}/hyps.json", "w"), ensure_ascii=False)
    if a.lm:
        ck = torch.load(a.lm); assert ck["itos"] == voc.itos, "LM and AM char vocabularies differ"
        lm = LM(len(voc.itos), ck["ctx"]).to(dev); lm.load_state_dict(ck["sd"]); lm.eval(); grid = {}
        for g in a.grid.split(","):
            al, be = map(float, g.split(":")); t0 = time.time()
            fn = lambda f, n, ml, al=al, be=be: beam_char(model, lm, ck["ctx"], voc, dev, f, n, ml, al, be, a.beam)
            md, _, _ = evaluate(model, fe, devset, voc, dev, beam_fn=fn, max_sec=60)
            grid[g] = md; print(f"alpha={al} beta={be}: dev WER {md['WER']:.2f} CER {md['CER']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        bg = min(grid, key=lambda k: grid[k]["WER"]); al, be = map(float, bg.split(":"))
        fn = lambda f, n, ml, **kwargs: beam_char(model, lm, ck["ctx"], voc, dev, f, n, ml, al, be, a.beam, **kwargs)
        lm_decoding_log = []
        ml_, rows_l, per_l = evaluate(model, fe, test, voc, dev, want=True, beam_fn=fn, max_sec=60,
                                     decoding_log=lm_decoding_log)
        ana_l = T.run_analysis(rows_l, train_sents, a.seed); ana_l.pop("WER_per_speaker", None)
        print("SELECTED after current dev evaluation:", bg, "TEST+LM", json.dumps(ml_), flush=True)
        res["lm"] = dict(best=bg, test=ml_, analysis=ana_l, dev_grid=grid, beam=a.beam, lm=a.lm,
                          selection_mode=("current_development_grid" if len(grid) > 1 else "single_configuration_dev_evaluation"),
                          decoding_log=lm_decoding_log)
        json.dump(per_l, open(f"{a.out}/hyps_lm.json", "w"), ensure_ascii=False)
    json.dump(res, open(f"{a.out}/result.json", "w"), ensure_ascii=False, indent=1)
    if not a.keep_ckpt: os.remove(f"{a.out}/best.pt")


if __name__ == "__main__":
    main()
