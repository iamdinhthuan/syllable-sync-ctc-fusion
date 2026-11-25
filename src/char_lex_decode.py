"""Character system control sharing intended syllable inventory, syllable LM and CTC prefix scoring.
Trie coverage, head multiplicity, sequence length, optimization and search/tuning still differ
from the structured (initial, rhyme, tone) system; this does not isolate representation alone.

Score of a hypothesis = (1-lam) * attention + lam * CTC prefix + alpha * syllable-LM + beta * #syllables.
The syllable-LM score is distributed over the characters of a syllable through look-ahead:
    inc(u -> v) = LA(v) - LA(u),  LA(node) = log sum_{w in subtree(node)} p_LM(w | history),  LA(root) := 0,
so the total for a finished syllable is exactly log p_LM(w | history).
"""
import sys, os, json, argparse, random, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_vivos as T
import train_char as TC
import tok
from train_vivos import PAD, BOS, EOS, UNK
from lm_model import LM, CTX
from ctc_prefix import ComponentCTC
from decoding_protocol import DecodeTracker

NEG = -1e9


class Trie:
    def __init__(self, words, cvoc, lm_stoi, V, dev):
        child = [[-1] * V]; term = [-1]; members = [[]]
        kept = 0
        for w in words:
            ids = [cvoc.xs[0].get(c) for c in w]
            if any(i is None for i in ids) or w not in lm_stoi: continue
            kept += 1; u = 0; members[0].append(lm_stoi[w])
            for c in ids:
                if child[u][c] < 0:
                    child[u][c] = len(child); child.append([-1] * V); term.append(-1); members.append([])
                u = child[u][c]; members[u].append(lm_stoi[w])
            term[u] = lm_stoi[w]
        self.child = torch.tensor(child, device=dev); self.term = torch.tensor(term, device=dev)
        rows = [n for n, m in enumerate(members) for _ in m]; cols = [x for m in members for x in m]
        self.M = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.ones(len(rows)), (len(child), len(lm_stoi_list(lm_stoi)))).coalesce().to(dev).to_sparse_csr()
        self.N = len(child); self.kept = kept


def lm_stoi_list(stoi):
    return [None] * (max(stoi.values()) + 1)


def lm_logits(lm, hist, dev):
    """hist: list of lists of LM ids (each starts with <s>=1). returns log-probs (R, V_lm) after the last token."""
    ctxs = [h[-CTX:] for h in hist]; L = max(len(h) for h in ctxs)
    x = torch.zeros(len(ctxs), L, dtype=torch.long, device=dev)
    for i, h in enumerate(ctxs): x[i, : len(h)] = torch.tensor(h, device=dev)
    out = lm(x).float()
    last = torch.tensor([len(h) - 1 for h in ctxs], device=dev)
    return F.log_softmax(out[torch.arange(len(ctxs), device=dev), last], -1)


@torch.no_grad()
def beam_lex(model, lm, trie, voc, dev, feats, n, max_len, alpha, beta, K, lam, sp, return_info=False):
    """Trie beam search; cap records retain the selected unclosed node and committed LM history."""
    mem, mm = model.encode(feats, n)
    position_buffer = getattr(getattr(model, "pe", None), "pe", None)
    if position_buffer is not None and (max_len is None or isinstance(max_len, int)):
        capacity = int(position_buffer.shape[-2])
        max_len = capacity if max_len is None else min(max_len, capacity)
    tracker = DecodeTracker(mm, max_len)
    B = feats.size(0); V = len(voc.itos)
    ctc = None
    if lam > 0:
        lens = (~mm).sum(1).repeat_interleave(K, 0)
        ctc = ComponentCTC(F.log_softmax(model.cheads[0](mem).float(), -1).repeat_interleave(K, 0), lens, blank=PAD)
    mem = mem.repeat_interleave(K, 0); mm = mm.repeat_interleave(K, 0)
    R = B * K; ar = torch.arange(R, device=dev)
    seq = torch.full((R, 1, 1), BOS, dtype=torch.long, device=dev)
    node = torch.zeros(R, dtype=torch.long, device=dev)
    hist = [[1] for _ in range(R)]
    score = torch.zeros(B, K, device=dev); score[:, 1:] = -1e9
    fin = torch.zeros(B, K, dtype=torch.bool, device=dev)
    for step in range(int(tracker.limits.max())):
        inactive = tracker.inactive(fin, step)
        att = F.log_softmax(model.decode(seq, mem, mm)[0][:, -1].float(), -1)            # R,V
        if ctc is not None:
            psi, end = ctc.scores(); cinc = (psi - ctc.psi[:, None]).float(); cinc[:, EOS] = (end - ctc.psi).float()
            am = (1 - lam) * att + lam * cinc
        else:
            am = att
        nxt = trie.child[node]                                                              # R,V
        valid = nxt >= 0
        lmlp = lm_logits(lm, hist, dev) if alpha > 0 or True else None                     # R,Vlm
        LA = torch.log(torch.sparse.mm(trie.M, lmlp.exp().t()).clamp_min(1e-30))           # N,R
        la_u = torch.where(node == 0, torch.zeros(R, device=dev), LA[node, ar])
        la_v = LA[nxt.clamp(min=0), ar[:, None]]                                            # R,V
        inc = torch.where(valid, la_v - la_u[:, None], torch.full_like(la_v, NEG))
        tw = trie.term[node]                                                                # word ending at the current node
        at_word = (tw >= 0) & (node != 0)
        p_w = lmlp[ar, tw.clamp(min=0)]
        # space closes the current syllable
        inc[:, sp] = torch.where(at_word, p_w - la_u, torch.full_like(p_w, NEG))
        # EOS: at the root (after a space) or right after a complete syllable
        hist_w = [h + [int(w)] if a_ else h for h, w, a_ in zip(hist, tw.tolist(), at_word.tolist())]
        p_end = lm_logits(lm, hist_w, dev)[:, 2]
        eos_inc = torch.where(at_word, p_w - la_u + p_end, torch.where(node == 0, lmlp[:, 2], torch.full_like(p_w, NEG)))
        started = seq.size(1) > 1
        if not started: eos_inc = torch.full_like(eos_inc, NEG)
        inc[:, EOS] = eos_inc
        bonus = torch.zeros(R, V, device=dev)
        bonus[:, sp] = beta; bonus[:, EOS] = torch.where(at_word, torch.full_like(p_w, beta), torch.zeros_like(p_w))
        add = am + alpha * torch.clamp(inc, min=NEG) + bonus
        add = torch.where(inc <= NEG / 2, torch.full_like(add, NEG), add)
        add[:, [PAD, BOS, UNK]] = NEG
        add = add.view(B, K, V)
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
        sel = ((torch.arange(B, device=dev) * K)[:, None] + par).view(-1); tkf = tk.view(-1)
        active = ~carried.view(-1)
        seq = torch.cat([seq[sel], tkf.view(-1, 1, 1)], 1)
        old_node, old_tw, old_at = node[sel], tw[sel], at_word[sel]
        new_node = trie.child[old_node, tkf].clamp(min=0)
        close = (tkf == sp) | (tkf == EOS)
        node = torch.where(active, torch.where(close, torch.zeros_like(new_node), new_node), old_node)
        hist = [hist[s] + ([int(w)] if (live and c and a_) else [])
                for s, w, c, a_, live in zip(sel.tolist(), old_tw.tolist(), close.tolist(), old_at.tolist(), active.tolist())]
        if ctc is not None: ctc.select(sel, tkf)
        fin = fin.gather(1, par) | ((tk == EOS) & ~carried)
        tracker.update(fin, step)
        if (fin | (step + 1 >= tracker.limits[:, None])).all(): break
    best = score.argmax(-1)
    out, records = tracker.finalize(seq.view(B, K, -1, 1)[:, :, 1:], fin, best, eos_id=EOS)
    for b, record in enumerate(records):
        selected = b * K + int(best[b])
        terminal = int(trie.term[node[selected]]) >= 0
        record.update(selected_score=float(score[b, best[b]]), trie_node=int(node[selected]),
                      lm_history=hist[selected], partial_word=bool(node[selected] != 0),
                      incomplete_syllable=bool(node[selected] != 0 and not terminal))
    return (out, records) if return_info else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="char run dir with best.pt and result.json")
    ap.add_argument("--chars", required=True); ap.add_argument("--lm", required=True)
    ap.add_argument("--beam", type=int, default=8); ap.add_argument("--root", default="")
    ap.add_argument("--grid", default="0:0.7:5,0.3:0.7:3")
    ap.add_argument("--tag", default="lex"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    args = json.load(open(f"{a.run}/result.json"))["args"]
    dev = torch.device("cuda"); random.seed(0)
    T.ROOT = a.root or args.get("root") or T.ROOT
    train_all, test = T.load_split("train"), T.load_split("test")
    spks = sorted({it["spk"] for it in train_all}); ds_ = set(spks[-3:])
    train = [it for it in train_all if it["spk"] not in ds_]; devset = [it for it in train_all if it["spk"] in ds_]
    if a.limit: devset, test = devset[: a.limit], test[: a.limit]
    voc = TC.CharVocab(json.load(open(a.chars, encoding="utf-8")))
    for part in (devset, test): TC.prep(part)
    train_sents = {" ".join(it["words"]) for it in train_all}
    fe = T.WavFrontend().to(dev)
    model = T.Model(voc, ssl=args["ssl"]).to(dev); model.load_state_dict(torch.load(f"{a.run}/best.pt")); model.eval()
    ck = torch.load(a.lm); itos = ck["itos"]; stoi = {w: i for i, w in enumerate(itos)}
    lm = LM(len(itos)).to(dev); lm.load_state_dict(ck["sd"]); lm.eval()
    # the SAME syllable inventory as decode_lm.py (LM vocabulary that the structured decoder can spell + train syllables)
    pv = T.Vocab(train, factor=False)
    words = set()
    for w in itos[4:]:
        p = tok.parse(w)
        if p and tok.roundtrip_ok(w) and T.UNK not in pv.enc([p])[0]: words.add(w)
    for it in train:
        for t in pv.enc(it["trip"]):
            if T.UNK not in t: words.add(tok.surface(*pv.dec(list(t))))
    trie = Trie(sorted(words), voc, stoi, len(voc.itos), dev)
    sp = voc.xs[0][" "]
    print(f"inventory {len(words)} kept-in-trie {trie.kept} nodes {trie.N}", flush=True)
    res = {}
    def run(items, lam, al, be, want=False, decoding_log=None):
        fn = lambda f, n, ml, **kwargs: beam_lex(model, lm, trie, voc, dev, f, n, ml, al, be, a.beam, lam, sp, **kwargs)
        return TC.evaluate(model, fe, items, voc, dev, want=want, beam_fn=fn, max_sec=60, decoding_log=decoding_log)
    for g in a.grid.split(","):
        lam, al, be = map(float, g.split(":")); t0 = time.time()
        md, _, _ = run(devset, lam, al, be)
        res[g] = md; print(f"ctc={lam} alpha={al} beta={be}: dev WER {md['WER']:.2f} CER {md['CER']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    bg = min(res, key=lambda k: res[k]["WER"]); lam, al, be = map(float, bg.split(":"))
    decoding_log = []
    mt, rows, per = run(test, lam, al, be, True, decoding_log=decoding_log)
    ana = T.run_analysis(rows, train_sents, 0); ana.pop("WER_per_speaker", None)
    print("SELECTED after current dev evaluation:", bg, "TEST", json.dumps(mt), flush=True)
    print("ANALYSIS", json.dumps(ana), flush=True)
    json.dump(dict(best=bg, test=mt, analysis=ana, dev_grid=res, beam=a.beam,
                   selection_mode=("current_development_grid" if len(res) > 1 else "single_configuration_dev_evaluation"),
                   decoding_log=decoding_log),
              open(f"{a.run}/result_{a.tag}.json", "w"), indent=1)
    json.dump(per, open(f"{a.run}/hyps_{a.tag}.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
