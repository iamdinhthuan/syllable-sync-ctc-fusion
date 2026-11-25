"""Character-level Transformer LM on cleaned Wikipedia + train transcripts (same text as the syllable LM)."""
import os, sys, json, math, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/ssd2/vispeech_exp")
import tok
from lm_model import LM
chars_json, wiki, train_prompts, out = sys.argv[1:5]
carve = os.environ.get("CARVE_DEV", "0") == "1"
itos = json.load(open(chars_json, encoding="utf-8")); V = len(itos); CTXC = 128
_cc = {}
def canon(w):
    c = _cc.get(w)
    if c is None:
        p = tok.parse(w); c = tok.surface(*p) if p else w; _cc[w] = c
    return c
lut = np.full(0x110000, 3, dtype=np.int16)
for i, c in enumerate(itos[4:], 4): lut[ord(c)] = i
lut[2] = 2; lut[3] = 1                                    # "\x02\x03" separates sentences: <eos><bos>
rows = [l.rstrip("\n").partition(" ") for l in open(train_prompts, encoding="utf-8")]
spk = sorted({r[0].split("_")[0] for r in rows}); dev = set(spk[-3:]) if carve else set()
tr = [" ".join(canon(w) for w in tok.norm_text(r[2]).split()) for r in rows if r[0].split("_")[0] not in dev]
wl = [" ".join(canon(w) for w in l.split()) for l in open(wiki, encoding="utf-8")]
text = "\x02\x03".join(wl + tr * 3)
ids = lut[np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)]
data = torch.from_numpy(np.concatenate([[1], ids, [2]]).astype(np.int16))
print("chars", V, "tokens", len(data), flush=True)
dev_ = "cuda"; lm = LM(V, CTXC).to(dev_)
BS = 128; H = 200000; N = len(data); steps = int(float(os.environ.get("EPOCHS", "1")) * N / (BS * CTXC))
opt = torch.optim.AdamW(lm.parameters(), 6e-4, betas=(0.9, 0.98), weight_decay=0.01)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
hold = data[:H].long()[: (H // (CTXC + 1)) * (CTXC + 1)].view(-1, CTXC + 1).to(dev_)[:128]
t0 = time.time()
for st in range(steps):
    ix = torch.randint(H, N - CTXC - 1, (BS,))
    x = torch.stack([data[i:i + CTXC + 1] for i in ix.tolist()]).long().to(dev_)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(lm(x[:, :-1]).float().reshape(-1, V), x[:, 1:].reshape(-1), ignore_index=0)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step(); sch.step()
    if st % 500 == 0 or st == steps - 1:
        lm.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hl = F.cross_entropy(lm(hold[:, :-1]).float().reshape(-1, V), hold[:, 1:].reshape(-1), ignore_index=0)
        lm.train(); print(f"step {st}/{steps} loss {loss.item():.3f} holdout bpc {hl.item()/math.log(2):.3f} {time.time()-t0:.0f}s", flush=True)
torch.save(dict(sd=lm.state_dict(), itos=itos, ctx=CTXC), out); print("saved", flush=True)
