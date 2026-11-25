import sys, glob, re, json, random
import pyarrow.parquet as pq
sys.path.insert(0, "/mnt/ssd2/vispeech_exp")
from tok import norm_text, parse
D = "/mnt/ssd2/vispeech_data"
test = set()
for l in open(f"{D}/vivos/vivos/test/prompts.txt", encoding="utf-8"):
    _, _, t = l.rstrip("\n").partition(" "); test.add(norm_text(t))
ok = {}
def good(t):
    w = t.split()
    if not (4 <= len(w) <= 40): return False
    for x in w:
        if x not in ok: ok[x] = parse(x) is not None
        if not ok[x]: return False
    return True
out = open("/mnt/ssd2/vispeech_exp/lm_text_wiki.txt", "w", encoding="utf-8")
seen = set(); n = skipped = leak = 0
for f in sorted(glob.glob(f"{D}/viwiki/20231101.vi/*.parquet")):
    pf = pq.ParquetFile(f)
    for b in pf.iter_batches(batch_size=2000, columns=["text"]):
        for txt in b.column("text").to_pylist():
            for para in txt.split("\n"):
                for s in re.split(r"(?<=[.!?;:])\s+", para):
                    t = norm_text(s)
                    if not good(t): skipped += 1; continue
                    if t in test: leak += 1; continue
                    if t in seen: continue
                    seen.add(t); out.write(t + "\n"); n += 1
    print(f, n, "skipped", skipped, "removed_test_dup", leak, flush=True)
print("DONE", n)
