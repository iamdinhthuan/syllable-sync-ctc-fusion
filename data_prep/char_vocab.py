"""Shared character vocabulary (specials + chars of canonical train transcripts + chars of the cleaned wiki text)."""
import sys, json, collections
sys.path.insert(0, "/mnt/ssd2/vispeech_exp")
import tok
ds, train_prompts, wiki, out = sys.argv[1:5]
_cc = {}
def canon(w):
    c = _cc.get(w)
    if c is None:
        p = tok.parse(w); c = tok.surface(*p) if p else w; _cc[w] = c
    return c
chars = collections.Counter()
for l in open(train_prompts, encoding="utf-8"):
    chars.update(" ".join(canon(w) for w in tok.norm_text(l.rstrip("\n").partition(" ")[2]).split()))
nt = set(chars)
wc = collections.Counter()
for l in open(wiki, encoding="utf-8"):
    wc.update(" ".join(canon(w) for w in l.split()))
itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(set(chars) | {c for c, n in wc.items() if n >= 50})
json.dump(itos, open(out, "w"), ensure_ascii=False)
print(ds, "chars", len(itos), "train-only chars", len(nt), "wiki-only", len(set(itos) - nt - set(itos[:4])))
