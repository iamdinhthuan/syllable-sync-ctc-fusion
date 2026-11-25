"""Syllable-level Transformer LM on Vietnamese Wikipedia text + ViVOS-train transcripts (no ViVOS test text)."""
import os, sys, json, math, time, collections, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/mnt/ssd2/vispeech_exp")
import tok
_cc = {}
def canon(w):                                   # same tone-mark placement as the ASR targets (old style), whatever the source used
    c = _cc.get(w)
    if c is None:
        p = tok.parse(w); c = tok.surface(*p) if p else w; _cc[w] = c
    return c
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ssd2/vispeech_exp/lm"
EPOCHS = float(sys.argv[2]) if len(sys.argv) > 2 else 2
R = os.environ.get("TRAIN_PROMPTS", "/mnt/ssd2/vispeech_data/vivos/vivos/train/prompts.txt")
WIKI = os.environ.get("WIKI", "/mnt/ssd2/vispeech_exp/lm_text_wiki_clean.txt")
CARVE = os.environ.get("CARVE_DEV", "1") == "1"      # ViVOS has no official dev split: its text is excluded for the 3 dev speakers
_rows = [l.rstrip("\n").partition(" ") for l in open(R, encoding="utf-8")]
_spk = sorted({r[0].split("_")[0] for r in _rows}); _dev = set(_spk[-3:]) if CARVE else set()      # same dev speakers as train_vivos.py: exclude their text
viv = [" ".join(canon(w) for w in tok.norm_text(r[2]).split()) for r in _rows if r[0].split("_")[0] not in _dev]
print("vivos LM sentences", len(viv), "excluded dev speakers", sorted(_dev), flush=True)
cnt = collections.Counter()
wiki = []
NO_WIKI = bool(os.environ.get("NO_WIKI"))          # ablation: LM from ViVOS-train transcripts only (no external text)
for l in ([] if NO_WIKI else open(WIKI, encoding="utf-8")):
    w = [canon(x) for x in l.split()]; wiki.append(w); cnt.update(w)
for s in viv: cnt.update(s.split())
vset = {w for s in viv for w in s.split()}
itos = ["<pad>", "<s>", "</s>", "<unk>"] + [w for w, c in cnt.most_common() if c >= 5 or w in vset]
stoi = {w: i for i, w in enumerate(itos)}
print("vocab", len(itos), "sentences", len(wiki), flush=True)
def enc(ws): return [1] + [stoi.get(w, 3) for w in ws] + [2]
stream = []
for ws in wiki: stream.extend(enc(ws))
for _ in range(3):
    for s in viv: stream.extend(enc(s.split()))
data = torch.tensor(stream, dtype=torch.int32)
print("tokens", len(data), flush=True)
V, D, L, CTX = len(itos), 512, 6, 64
class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, D); s.pos = nn.Embedding(CTX, D)
        layer = nn.TransformerEncoderLayer(D, 8, 2048, 0.1, batch_first=True, norm_first=True)
        s.tr = nn.TransformerEncoder(layer, L, norm=nn.LayerNorm(D), enable_nested_tensor=False)
        s.out = nn.Linear(D, V, bias=False); s.out.weight = s.emb.weight
        nn.init.normal_(s.emb.weight, std=D ** -0.5); nn.init.normal_(s.pos.weight, std=0.02)
    def forward(s, x):
        T = x.size(1)
        h = s.emb(x) + s.pos(torch.arange(T, device=x.device))
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        return s.out(s.tr(h, mask=m, is_causal=True))
if __name__ == "__main__":
    dev = "cuda"; lm = LM().to(dev)
    print("params %.1fM" % (sum(p.numel() for p in lm.parameters()) / 1e6), flush=True)
    BS = 192; steps = int(EPOCHS * len(data) / (BS * CTX))
    opt = torch.optim.AdamW(lm.parameters(), 6e-4, betas=(0.9, 0.98), weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    # hold-out: last 20000 tokens of wiki-part (approx) for ppl; ViVOS dev measured in decode script
    H = 20000 if NO_WIKI else 200000
    hold = data[:H].long(); N = len(data); t0 = time.time()
    for st in range(steps):
        ix = torch.randint(H, N - CTX - 1, (BS,))
        x = torch.stack([data[i:i + CTX + 1] for i in ix.tolist()]).long().to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(lm(x[:, :-1]).float().reshape(-1, V), x[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step(); sch.step()
        if st % 200 == 0 or st == steps - 1:
            lm.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hx = hold[: (len(hold) // (CTX + 1)) * (CTX + 1)].view(-1, CTX + 1).to(dev)[:256]
                hl = F.cross_entropy(lm(hx[:, :-1]).float().reshape(-1, V), hx[:, 1:].reshape(-1), ignore_index=0)
            lm.train(); print(f"step {st}/{steps} loss {loss.item():.3f} holdout ppl {math.exp(hl.item()):.1f} {time.time()-t0:.0f}s", flush=True)
    torch.save(dict(sd=lm.state_dict(), itos=itos), OUT + ".pt"); print("saved", flush=True)
