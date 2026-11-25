import sys
sys.path.insert(0, "/mnt/ssd2/vispeech_exp")
import tok
N = 5
g = set()
for tp in sys.argv[2:]:                                   # usage: clean_corpus.py OUT test_prompts.txt [more test files]
    for l in open(tp, encoding="utf-8"):
        w = tok.norm_text(l.rstrip("\n").partition(" ")[2]).split()
        g.update(tuple(w[j:j + N]) for j in range(len(w) - N + 1))
kept = drop = 0
with open(sys.argv[1], "w", encoding="utf-8") as out:
    for l in open("lm_text_wiki.txt", encoding="utf-8"):
        w = l.split()
        if any(tuple(w[j:j + N]) in g for j in range(len(w) - N + 1)): drop += 1; continue
        out.write(l); kept += 1
print("kept", kept, "dropped", drop)
