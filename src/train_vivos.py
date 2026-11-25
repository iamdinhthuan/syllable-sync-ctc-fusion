"""E0: from-scratch phonemic (initial, rhyme, tone) ASR on ViVOS, re-implemented from the ViSpeechFormer description.

Encoder : conv2d/4 subsampling + N transformer layers   (paper: 4 layers)
Decoder : 1 transformer decoder layer; each step emits one syllable as 3 heads (initial, rhyme, tone)
Eval    : CER, WER (whitespace syllables), per-component PER; seen/unseen sentence split; speaker-cluster bootstrap.
"""
import argparse, collections, json, math, os, random, re, sys, time
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
# cuDNN SDPA caches an execution plan per input shape in host memory; with variable-length audio this grows without bound (pod OOM). Use flash / mem-efficient attention instead.
torch.backends.cuda.enable_cudnn_sdp(False)
from rapidfuzz.distance import Levenshtein as Lev
from decoding_protocol import DecodeTracker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tok

ROOT = "/mnt/ssd2/vispeech_data/vivos/vivos"
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = 0, 1, 2, 3


# ----------------------------------------------------------------------------- data
def load_split(split, limit=None):
    items = []
    for line in open(f"{ROOT}/{split}/prompts.txt", encoding="utf-8"):
        uid, _, text = line.rstrip("\n").partition(" ")
        spk = uid.split("_")[0]
        wav = f"{ROOT}/{split}/waves/{spk}/{uid}.wav"
        words = tok.norm_text(text).split()
        trip = [tok.parse(w) if tok.roundtrip_ok(w) else None for w in words]
        canon = [tok.surface(*t) if t else w for w, t in zip(words, trip)]   # canonical spelling of the reference
        items.append(dict(uid=uid, spk=spk, wav=wav, words=canon, trip=trip))
    if limit:
        items = items[:limit]
    return items


def read_wav(path):
    x, sr = sf.read(path, dtype="int16")
    assert sr == 16000, sr
    return x


def split_rhyme(r):
    """rhyme -> (nucleus incl. glide, final); always nuc + fin == r (bijective)."""
    if r == "uy":
        return r, ""
    m = re.match(r"^(.*?)(ch|nh|ng|c|m|n|p|t|i|y|o|u)?$", r)
    nuc, fin = m.group(1), m.group(2) or ""
    return (r, "") if not nuc else (nuc, fin)


class Vocab:
    """Components per syllable: (initial, rhyme, tone) or, with factor=True, (initial, nucleus, final, tone)."""
    def __init__(self, items, factor=False):
        self.factor = factor
        ci, cr, cn, cf = set(), set(), set(), set()
        for it in items:
            for t in it["trip"]:
                if t:
                    ci.add(t[0]); cr.add(t[1])
                    n_, f_ = split_rhyme(t[1]); cn.add(n_); cf.add(f_ or "∅f")
        self.i = SPECIAL + sorted(ci)
        self.r = SPECIAL + sorted(cr)
        self.t = SPECIAL + tok.TONE_LIST
        self.n = SPECIAL + sorted(cn)
        self.f = SPECIAL + sorted(cf)
        self.vocs = [self.i, self.n, self.f, self.t] if factor else [self.i, self.r, self.t]
        self.xs = [{x: k for k, x in enumerate(v)} for v in self.vocs]
        self.nc = len(self.vocs)

    def comp_strs(self, t):
        if not self.factor:
            return t
        n_, f_ = split_rhyme(t[1])
        return (t[0], n_, f_ or "∅f", t[2])

    def enc(self, trip):
        out = []
        for t in trip:
            if t is None:
                out.append((UNK,) * self.nc)
            else:
                cs = self.comp_strs(t)
                out.append(tuple(self.xs[j].get(c, UNK) for j, c in enumerate(cs)))
        return out

    def dec(self, ids):
        """ids -> (initial, rhyme, tone) strings."""
        cs = [self.vocs[j][k] for j, k in enumerate(ids)]
        if not self.factor:
            return tuple(cs)
        return cs[0], cs[1] + ("" if cs[2] == "∅f" else cs[2]), cs[3]


def batches(items, max_sec, shuffle, rng):
    order = sorted(range(len(items)), key=lambda k: items[k]["dur"])
    out, cur, mx = [], [], 0.0
    for k in order:
        d = items[k]["dur"]
        if cur and max(mx, d) * (len(cur) + 1) > max_sec:
            out.append(cur); cur, mx = [], 0.0
        cur.append(k); mx = max(mx, d)
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


# ----------------------------------------------------------------------------- features
def mel_basis(n_mels=80, n_fft=512, sr=16000, fmin=20.0, fmax=7600.0):
    def hz2mel(f): return 2595 * math.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = torch.tensor([mel2hz(m) for m in np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)])
    bins = torch.floor((n_fft + 1) * pts / sr).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fb", mel_basis())
        self.register_buffer("win", torch.hann_window(400))

    @torch.no_grad()
    def forward(self, wav, wav_len):
        x = wav.float() / 32768.0
        x = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], 1)
        s = torch.stft(x, 512, 160, 400, self.win, return_complex=True).abs() ** 2
        m = torch.log(torch.matmul(self.fb, s) + 1e-6).transpose(1, 2)           # B,T,80
        n = (wav_len // 160) + 1
        mask = torch.arange(m.size(1), device=m.device)[None] < n[:, None]
        mean = (m * mask[..., None]).sum(1, keepdim=True) / n[:, None, None]
        var = (((m - mean) ** 2) * mask[..., None]).sum(1, keepdim=True) / n[:, None, None]
        return (m - mean) / (var.sqrt() + 1e-5) * mask[..., None], n


def spec_augment(x, n):
    B, T, Fq = x.shape
    for b in range(B):
        for _ in range(2):
            f = random.randint(0, 15); f0 = random.randint(0, Fq - f)
            x[b, :, f0:f0 + f] = 0
        for _ in range(2):
            t = random.randint(0, min(40, int(n[b]) // 5)); t0 = random.randint(0, max(int(n[b]) - t, 0))
            x[b, t0:t0 + t, :] = 0
    return x


class WavFrontend(nn.Module):
    """Per-utterance mean/var normalisation of the raw waveform (what wav2vec2/HuBERT expect)."""
    @torch.no_grad()
    def forward(self, wav, wav_len):
        x = wav.float() / 32768.0
        mask = torch.arange(x.size(1), device=x.device)[None] < wav_len[:, None]
        n = wav_len[:, None].float()
        mean = (x * mask).sum(1, keepdim=True) / n
        var = (((x - mean) ** 2) * mask).sum(1, keepdim=True) / n
        return (x - mean) / (var.sqrt() + 1e-7) * mask, wav_len


# ----------------------------------------------------------------------------- model
class PosEnc(nn.Module):
    def __init__(self, d, maxlen=4000):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class Model(nn.Module):
    def __init__(self, voc, d=256, enc_layers=4, dec_layers=1, heads=4, ff=1024, drop=0.1, ssl=""):
        super().__init__()
        self.voc = voc
        self.ssl = None
        if ssl:
            from transformers import AutoModel
            self.ssl = AutoModel.from_pretrained(ssl, mask_time_prob=0.05, mask_time_length=5,
                                                 layerdrop=0.0, apply_spec_augment=True)
            self.ssl.feature_extractor._freeze_parameters() if hasattr(self.ssl, "feature_extractor") else None
            if os.environ.get("SSL_GC"): self.ssl.gradient_checkpointing_enable()          # memory for the 1B encoder
            self.ssl_proj = nn.Linear(self.ssl.config.hidden_size, d)
        cc = 64
        self.conv = nn.Sequential(nn.Conv2d(1, cc, 3, 2), nn.ReLU(), nn.Conv2d(cc, cc, 3, 2), nn.ReLU())
        self.proj = nn.Linear(cc * 19, d)
        self.pe = PosEnc(d)
        self.drop = nn.Dropout(drop)
        el = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.emb = nn.ModuleList(nn.Embedding(len(v), d) for v in voc.vocs)
        dl = nn.TransformerDecoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec_layers, norm=nn.LayerNorm(d))
        self.heads = nn.ModuleList(nn.Linear(d, len(v)) for v in voc.vocs)
        self.cheads = nn.ModuleList(nn.Linear(d, len(v)) for v in voc.vocs)      # auxiliary frame-level CTC heads

    def encode(self, feats, n):
        if self.ssl is not None:                                            # feats = normalized waveform (B,T), n = samples
            att = (torch.arange(feats.size(1), device=feats.device)[None] < n[:, None]).long()
            x = self.ssl(feats, attention_mask=att).last_hidden_state
            fl = self.ssl._get_feat_extract_output_lengths(n).clamp(max=x.size(1))
            mask = torch.arange(x.size(1), device=x.device)[None] >= fl[:, None]
            return self.drop(self.ssl_proj(x)), mask
        x = self.conv(feats.unsqueeze(1))                                   # B,d,T',19
        B, C, T, Fq = x.shape
        x = self.proj(x.permute(0, 2, 1, 3).reshape(B, T, C * Fq))
        n = ((n - 3) // 2 + 1 - 3) // 2 + 1
        n = n.clamp(max=T)
        mask = torch.arange(T, device=x.device)[None] >= n[:, None]         # True = pad
        x = self.enc(self.drop(self.pe(x)), src_key_padding_mask=mask)
        return x, mask

    def decode(self, inp, mem, mem_mask):
        L = inp.size(1)
        y = sum(e(inp[..., j]) for j, e in enumerate(self.emb))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=inp.device), 1)
        h = self.dec(self.drop(self.pe(y)), mem, tgt_mask=causal, memory_key_padding_mask=mem_mask)
        return tuple(hd(h) for hd in self.heads)


def ctc_loss(model, mem, mm, tgt):
    """Frame-level CTC on the encoder output, one head per component (blank = PAD id 0)."""
    in_len = (~mm).sum(1)
    tg = tgt[:, 1:]                                           # drop BOS
    lens = ((tg[..., 0] != PAD) & (tg[..., 0] != EOS)).sum(1)
    total = 0.0
    for j, head in enumerate(model.cheads):
        lp = F.log_softmax(head(mem).float(), -1).transpose(0, 1)
        flat = torch.cat([tg[b, : lens[b], j] for b in range(tg.size(0))])
        total = total + F.ctc_loss(lp, flat, in_len, lens, blank=PAD, zero_infinity=True)
    return total


# ----------------------------------------------------------------------------- train / eval
def collate_audio(items, idx, device):
    """Pad waveform inputs for inference without touching transcript fields."""
    wavs = [items[k]["audio"] for k in idx]
    wl = np.array([len(w) for w in wavs], dtype=np.int64)
    wav = np.zeros((len(wavs), int(wl.max())), dtype=np.int16)
    for b, w in enumerate(wavs):
        wav[b, : len(w)] = w
    return torch.from_numpy(wav).to(device), torch.from_numpy(wl).to(device)


def collate(items, idx, voc, device, train):
    wav, wl = collate_audio(items, idx, device)
    seqs = [[(BOS,) * voc.nc] + voc.enc(items[k]["trip"]) + [(EOS,) * voc.nc] for k in idx]
    tgt = np.zeros((len(seqs), max(len(s) for s in seqs), voc.nc), dtype=np.int64)
    for b, s in enumerate(seqs):
        tgt[b, : len(s)] = s
    return wav, wl, torch.from_numpy(tgt).to(device)


@torch.no_grad()
def greedy(model, feats, n, max_len=None, allowed=None, lam=None, return_info=False):
    """Greedy decoding. If `allowed` (A,NC) is given, each step picks the best syllable among `allowed`
    (heads are conditionally independent, so the exact argmax is a gather); an unconstrained syllable may still win
    if its score beats the best allowed one by more than `lam` (lam=inf: hard constraint).
    Each row's valid encoder frames plus EOS is bounded by decoder positional capacity and any max_len guard.
    return_info=True adds per-row natural-EOS/cap records without changing the default list API."""
    mem, mm = model.encode(feats, n)
    position_buffer = getattr(getattr(model, "pe", None), "pe", None)
    if position_buffer is not None and (max_len is None or isinstance(max_len, int)):
        capacity = int(position_buffer.shape[-2])  # [positions,d] or [1,positions,d]
        max_len = capacity if max_len is None else min(max_len, capacity)
    tracker = DecodeTracker(mm, max_len)
    B, NC = feats.size(0), model.voc.nc
    seq = torch.full((B, 1, NC), BOS, dtype=torch.long, device=feats.device)
    done = torch.zeros(B, dtype=torch.bool, device=feats.device)
    for step in range(int(tracker.limits.max())):
        inactive = tracker.inactive(done, step)
        lps = [F.log_softmax(x[:, -1].float(), -1) for x in model.decode(seq, mem, mm)]
        free = torch.stack([lp.argmax(-1) for lp in lps], -1)
        if allowed is not None:
            sc = sum(lp[:, allowed[:, j]] for j, lp in enumerate(lps))                       # B,A
            best, k = sc.max(-1)
            cons = allowed[k]
            if lam is None or lam == float("inf"):
                nxt = cons
            else:
                free_sc = sum(lp.max(-1).values for lp in lps)
                nxt = torch.where((free_sc - best > lam)[:, None], free, cons)
        else:
            nxt = free
        nxt = torch.where(inactive[:, None], torch.full_like(nxt, PAD), nxt)
        seq = torch.cat([seq, nxt[:, None]], 1)
        done |= (nxt == EOS).any(-1) & ~inactive
        tracker.update(done, step)
        if (done | (step + 1 >= tracker.limits)).all():
            break
    out, records = tracker.finalize(seq[:, None, 1:], done[:, None], eos_id=EOS)
    return (out, records) if return_info else out


def ids_to_words(seq, voc):
    words, comps = [], []
    for ids in seq:
        if EOS in ids:
            break
        if PAD in ids or BOS in ids:
            continue
        if UNK in ids:
            words.append("<unk>"); comps.append(("<unk>",) * 3)
        else:
            ini, rh, to = voc.dec(ids)
            words.append(tok.surface(ini, rh, to)); comps.append((ini, rh, to))
    return words, comps


def evaluate(model, frontend, items, voc, device, max_sec=200, want_hyps=False, allowed=None, lam=None, decoding_log=None):
    model.eval()
    hyps = {}
    for idx in batches(items, max_sec, False, None):
        wav, wl = collate_audio(items, idx, device)
        with torch.autocast(wav.device.type, dtype=torch.bfloat16, enabled=wav.is_cuda):
            feats, n = frontend(wav, wl)
            result = greedy(model, feats, n, max_len=None, allowed=allowed, lam=lam,
                            return_info=decoding_log is not None)
        if decoding_log is not None:
            out, records = result
            decoding_log.extend(dict(uid=items[k]["uid"], **record) for k, record in zip(idx, records))
        else:
            out = result
        for k, seq in zip(idx, out):
            hyps[k] = ids_to_words(seq, voc)
    return score(items, hyps, want_hyps)


def score(items, hyps, want_hyps=False):
    tot = collections.Counter()
    per = {}
    rows = []
    for k, it in enumerate(items):
        hw, hc = hyps[k]
        rw = it["words"]
        ew = Lev.distance(rw, hw)
        ec = Lev.distance(" ".join(rw), " ".join(hw))
        rc = [t if t else ("<unk>",) * 3 for t in it["trip"]]
        e = [Lev.distance([c[j] for c in rc], [c[j] for c in hc]) for j in range(3)]
        n = len(rw)
        tot.update(ew=ew, nw=n, ec=ec, nc=len(" ".join(rw)), ei=e[0], er=e[1], et=e[2])
        rows.append((it["spk"], it["sent"], len(rw), ew, ec, len(" ".join(rw)), *e))
        if want_hyps:
            per[it["uid"]] = dict(ref=" ".join(rw), hyp=" ".join(hw))
    m = dict(WER=100 * tot["ew"] / tot["nw"], CER=100 * tot["ec"] / tot["nc"],
             PER_i=100 * tot["ei"] / tot["nw"], PER_r=100 * tot["er"] / tot["nw"], PER_t=100 * tot["et"] / tot["nw"])
    m["PER"] = (m["PER_i"] + m["PER_r"] + m["PER_t"]) / 3
    return m, rows, per


def run_analysis(rows, train_sents, seed=0):
    """seen/unseen sentence split, length bins, speaker-cluster bootstrap CI for WER."""
    A = np.array([(r[2], r[3], r[5], r[4]) for r in rows])
    seen = np.array([r[1] in train_sents for r in rows])
    def wer(mask): return 100 * A[mask, 1].sum() / max(A[mask, 0].sum(), 1)
    out = dict(WER_seen=wer(seen), WER_unseen=wer(~seen), n_seen=int(seen.sum()), n_unseen=int((~seen).sum()))
    lens = A[:, 0]
    for lo, hi in ((0, 8), (8, 12), (12, 16), (16, 99)):
        mk = (lens >= lo) & (lens < hi)
        out[f"WER_len[{lo},{hi})"] = (wer(mk), int(mk.sum()))
    spk = collections.defaultdict(list)
    for r in rows:
        spk[r[0]].append(r)
    keys = sorted(spk)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(1000):
        pick = rng.choice(len(keys), len(keys))
        e = sum(sum(x[3] for x in spk[keys[p]]) for p in pick); n = sum(sum(x[2] for x in spk[keys[p]]) for p in pick)
        bs.append(100 * e / n)
    out["WER_ci95_speaker_bootstrap"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["WER_per_speaker"] = {k: 100 * sum(x[3] for x in v) / sum(x[2] for x in v) for k, v in spk.items()}
    return out


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--ssl", default="", help="HF id/path of an SSL encoder (replaces the from-scratch encoder)")
    ap.add_argument("--ssl_lr", type=float, default=5e-5)
    ap.add_argument("--factor_rhyme", action="store_true", help="E2: decoder/CTC predict nucleus and final instead of the whole rhyme")
    ap.add_argument("--max_sec", type=float, default=150.0)
    ap.add_argument("--keep_ckpt", action="store_true", help="keep best.pt after the run (default: delete to save disk)")
    ap.add_argument("--ctc_w", type=float, default=0.0, help="weight of auxiliary frame-level CTC (0 = pure attention decoder)")
    ap.add_argument("--root", default=ROOT, help="dataset root with {train,test[,dev]}/prompts.txt + waves/<spk>/<uid>.wav")
    ap.add_argument("--dev_split", default="", help="use this split as dev (default: carve the last 3 train speakers)")
    ap.add_argument("--dev_max", type=int, default=0, help="subsample the dev set to N utterances (seeded)")
    ap.add_argument("--init", default="", help="initialise from this state_dict (continue training)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N utterances per split")
    ap.add_argument("--train_frac", type=float, default=1.0, help="low-resource curve: keep this fraction of training utterances (dev/test untouched; label inventory from full train)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda")

    ROOT = a.root
    train_all, test = load_split("train"), load_split("test")
    if a.dev_split:
        train, devset = train_all, load_split(a.dev_split)
    else:
        spks = sorted({it["spk"] for it in train_all})
        dev_spk = set(spks[-3:])                                   # speaker-disjoint dev set
        train = [it for it in train_all if it["spk"] not in dev_spk]
        devset = [it for it in train_all if it["spk"] in dev_spk]
    if a.dev_max and len(devset) > a.dev_max:
        random.Random(0).shuffle(devset); devset = devset[: a.dev_max]
    if a.limit:                                                    # smoke test: subsample after the speaker split
        random.Random(0).shuffle(train); train = train[: a.limit]
        devset = devset[: max(a.limit // 5, 10)]; test = test[: max(a.limit // 5, 10)]
    voc = Vocab(train, factor=a.factor_rhyme)
    if a.train_frac < 1:
        random.Random(0).shuffle(train); train = train[: int(len(train) * a.train_frac)]; print('train_frac', a.train_frac, 'utts', len(train), flush=True)
    train_sents = {" ".join(it["words"]) for it in train_all}
    for it in train + devset + test:
        it["audio"] = read_wav(it["wav"]); it["dur"] = len(it["audio"]) / 16000
        it["sent"] = " ".join(it["words"])
    print(f"train {len(train)} dev {len(devset)} test {len(test)} | components={[len(v) for v in voc.vocs]}", flush=True)

    fe = (WavFrontend() if a.ssl else Frontend()).to(dev)
    model = Model(voc, enc_layers=a.enc_layers, ssl=a.ssl).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init)); print("init from", a.init, flush=True)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6, flush=True)
    ssl_p = [p for n_, p in model.named_parameters() if n_.startswith("ssl.") and p.requires_grad]
    oth_p = [p for n_, p in model.named_parameters() if not n_.startswith("ssl.")]
    groups = [dict(params=oth_p, lr=a.lr)] + ([dict(params=ssl_p, lr=a.ssl_lr)] if ssl_p else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=0.01)
    steps_per_epoch = len(batches(train, a.max_sec, False, None))
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))))
    rng = random.Random(a.seed)
    best, bad, step, log = 1e9, 0, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tl = 0.0; nb = 0
        for idx in batches(train, a.max_sec, True, rng):
            wav, wl, tgt = collate(train, idx, voc, dev, True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats, n = fe(wav, wl)
                if not a.ssl:
                    feats = spec_augment(feats, n)
                mem, mm = model.encode(feats, n)
                logits = model.decode(tgt[:, :-1], mem, mm)
                y = tgt[:, 1:]
                loss = sum(F.cross_entropy(l.float().reshape(-1, l.size(-1)), y[..., j].reshape(-1), ignore_index=PAD, label_smoothing=0.1)
                           for j, l in enumerate(logits))
                if a.ctc_w > 0:
                    loss = loss + a.ctc_w * ctc_loss(model, mem, mm, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); step += 1; tl += float(loss); nb += 1
        m, _, _ = evaluate(model, fe, devset, voc, dev)
        log.append(dict(epoch=ep, loss=tl / nb, **{f"dev_{k}": v for k, v in m.items()}))
        print(f"ep {ep:3d} loss {tl/nb:.3f} dev CER {m['CER']:.2f} WER {m['WER']:.2f} PER {m['PER']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if m["CER"] < best - 1e-4:
            best, bad = m["CER"], 0
            torch.save(model.state_dict(), f"{a.out}/best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop", flush=True); break
    model.load_state_dict(torch.load(f"{a.out}/best.pt"))
    decoding_log = []
    m, rows, per = evaluate(model, fe, test, voc, dev, want_hyps=True, decoding_log=decoding_log)
    ana = run_analysis(rows, train_sents, a.seed)
    # ---- lexicon-constrained decoding (E3): allowed syllables = train-split vocabulary only
    trip = {t for it in train for t in voc.enc(it["trip"])} | {(EOS,) * voc.nc}
    allowed = torch.tensor(sorted(trip), device=dev)
    cons = {}
    for lam in (float("inf"), 8.0, 5.0, 3.0, 1.5):
        md, _, _ = evaluate(model, fe, devset, voc, dev, allowed=allowed, lam=lam)
        constrained_log = []
        mt, rows_c, per_c = evaluate(model, fe, test, voc, dev, want_hyps=True, allowed=allowed, lam=lam,
                                    decoding_log=constrained_log)
        cons[str(lam)] = dict(dev=md, test=mt, rows=rows_c, per=per_c, decoding_log=constrained_log)
        print(f"constrained lam={lam}: dev WER {md['WER']:.2f} | test WER {mt['WER']:.2f} CER {mt['CER']:.2f} PER {mt['PER']:.2f}", flush=True)
    best_lam = min(cons, key=lambda k: cons[k]["dev"]["WER"])                  # chosen on dev only
    m_c = cons[best_lam]["test"]
    ana_c = run_analysis(cons[best_lam]["rows"], train_sents, a.seed)
    json.dump(per, open(f"{a.out}/hyps_unconstrained.json", "w"), ensure_ascii=False)
    per = cons[best_lam]["per"]
    constrained = dict(best_lam=best_lam, test=m_c, analysis={k: v for k, v in ana_c.items() if k != "WER_per_speaker"},
                       selection_mode="current_development_grid",
                       decoding_log=cons[best_lam]["decoding_log"],
                       all={k: dict(dev=v["dev"], test=v["test"], decoding_log=v["decoding_log"]) for k, v in cons.items()})
    json.dump(dict(args=vars(a), test=m, analysis=ana, constrained=constrained, log=log,
                   decoding_log=decoding_log, vocab=[len(v) for v in voc.vocs]),
              open(f"{a.out}/result.json", "w"), ensure_ascii=False, indent=1)
    json.dump(per, open(f"{a.out}/hyps.json", "w"), ensure_ascii=False)
    if not a.keep_ckpt:
        os.remove(f"{a.out}/best.pt")
    print("TEST", json.dumps(m), flush=True)
    print("ANALYSIS", json.dumps({k: v for k, v in ana.items() if k != "WER_per_speaker"}), flush=True)


if __name__ == "__main__":
    main()
