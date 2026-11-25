"""Convert LSVSC / ViMD parquet shards to the ViVOS-style layout (16 kHz mono wav + prompts.txt + meta.json)."""
import sys, os, io, json, glob, re
from multiprocessing import Pool
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from scipy.signal import resample_poly
from math import gcd
SRC = {"lsvsc": "/mnt/ssd2/vispeech_data/LSVSC/data", "vimd": "/mnt/ssd2/vispeech_data/ViMD_Dataset/data"}
SPLIT = {"train": "train", "validation": "dev", "valid": "dev", "test": "test"}
DST = "/mnt/ssd2/vispeech_data/prep"

def work(job):
    ds, f, split, shard = job
    pf = pq.ParquetFile(f); rows = []
    k = 0
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg).to_pylist():
            if ds == "lsvsc":
                grp = f"{split[:2]}{shard:02d}{k // 100:03d}"            # no speaker ids in LSVSC: group = block of 100 consecutive utts
                uid = f"{grp}_{shard:02d}{k:05d}"; text = r["transcription"]
                meta = dict(dialect=r["dialect"], gender=r["gender"], age=r["age"], topic=r["topic"], emotion=r["emotion"])
            else:
                grp = r["speakerID"].replace("_", "-"); stem = os.path.splitext(r["filename"])[0].replace("_", "-")
                uid = f"{grp}_{stem}"; text = r["text"]
                meta = dict(region=r["region"], province=r["province_name"], gender=r["gender"])
            k += 1
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if x.ndim > 1: x = x.mean(1)
            if sr != 16000:
                g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g)
            x = np.clip(x, -1, 1)
            d = f"{DST}/{ds}/{split}/waves/{grp}"; os.makedirs(d, exist_ok=True)
            sf.write(f"{d}/{uid}.wav", (x * 32767).astype(np.int16), 16000, subtype="PCM_16")
            rows.append((uid, " ".join(text.split()), meta, len(x) / 16000))
    return split, rows

if __name__ == "__main__":
    ds = sys.argv[1]; jobs = []
    for f in sorted(glob.glob(f"{SRC[ds]}/*.parquet")):
        name = os.path.basename(f); sp = SPLIT[name.split("-")[0]]
        jobs.append((ds, f, sp, int(re.search(r"-(\d+)-of", name).group(1))))
    out = {}
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as p:
        for i, (split, rows) in enumerate(p.imap_unordered(work, jobs)):
            out.setdefault(split, []).extend(rows); print(i + 1, "/", len(jobs), split, len(rows), flush=True)
    for split, rows in out.items():
        rows.sort()
        os.makedirs(f"{DST}/{ds}/{split}", exist_ok=True)
        with open(f"{DST}/{ds}/{split}/prompts.txt", "w", encoding="utf-8") as fo:
            for uid, text, _, _ in rows: fo.write(f"{uid} {text}\n")
        json.dump({u: dict(m, dur=d) for u, _, m, d in rows}, open(f"{DST}/{ds}/{split}/meta.json", "w"), ensure_ascii=False)
        print(ds, split, len(rows), "utts", round(sum(r[3] for r in rows) / 3600, 2), "h", flush=True)
