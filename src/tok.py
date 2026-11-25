"""Vietnamese syllable <-> (initial, rhyme, tone) triplets (orthographic ViPhonER-style)."""
import re
import unicodedata

TONES = {"̀": "huyen", "́": "sac", "̉": "hoi", "̃": "nga", "̣": "nang"}
TONE_MARK = {v: k for k, v in TONES.items()}
TONE_LIST = ["ngang", "sac", "huyen", "hoi", "nga", "nang"]
ONSETS = ["ngh", "ng", "nh", "ph", "th", "tr", "ch", "kh", "gh", "gi", "qu",
          "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "x"]
VOWELS = set("aăâeêioôơuưy")
MODIFIED = set("ăâêôơư")
NEW_STYLE = False  # tone-mark placement convention; False = old style (matches ViVOS)
RHYME_RE =re.compile(r"^[aăâeêioôơuưy]+(ch|nh|ng|c|m|n|p|t|i|y|o|u)?$")


def norm_text(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)          # drop punctuation
    s = re.sub(r"_", " ", s)
    return " ".join(s.split())


def split_tone(tok):
    tone = "ngang"
    out = []
    for ch in unicodedata.normalize("NFD", tok):
        if ch in TONES:
            tone = TONES[ch]
        else:
            out.append(ch)
    return tone, unicodedata.normalize("NFC", "".join(out))


def parse(tok):
    """(initial, rhyme, tone) or None."""
    tone, base = split_tone(tok)
    if not base or not base.isalpha() or any(c in "fjwz" for c in base):
        return None
    init, rest = "", base
    for o in ONSETS:
        if base.startswith(o):
            init, rest = o, base[len(o):]
            break
    if init == "gi":
        if not rest or rest[0] not in VOWELS:
            rest = "i" + rest
    elif init == "qu":
        init, rest = "q", "u" + rest
    if not RHYME_RE.match(rest):
        return None
    return init or "∅", rest, tone


def _place(init, rhyme, tone):
    if tone == "ngang":
        return rhyme
    chars = list(rhyme)
    start = 1 if init == "q" else 0           # 'u' after q is a glide, never takes the mark
    vs = [i for i, c in enumerate(chars) if c in VOWELS and i >= start]
    if not vs:                                  # ill-formed combination (e.g. q + 'u'): never raise
        vs = [i for i, c in enumerate(chars) if c in VOWELS]
        if not vs:
            return rhyme
    mod = [i for i in vs if chars[i] in MODIFIED]
    if mod:
        pos = mod[-1]
    elif len(vs) == 1:
        pos = vs[0]
    else:
        has_final = vs[-1] != len(chars) - 1
        pair = "".join(chars[i] for i in vs)
        if len(vs) == 2 and not has_final:
            # ViVOS uses the old spelling (hóa, hòa, khỏe, thủy): mark on the first vowel.
            pos = vs[1] if (NEW_STYLE and pair in ("oa", "oe", "uy")) else vs[0]
        else:
            pos = vs[1]
    chars[pos] = unicodedata.normalize("NFC", chars[pos] + TONE_MARK[tone])
    return "".join(chars)


def surface(init, rhyme, tone):
    """Inverse of parse (handles the 'gì' / 'qu' special cases)."""
    if init == "∅":
        return _place(init, rhyme, tone)
    if init == "gi":
        # parse() prepends 'i' when the remainder after 'gi' is not a vowel-initial: gì, gìn, gỉ ...
        # such rhymes start with 'i'; ambiguous with real gi+i... : treat leading 'i' rhyme as g+rhyme
        if rhyme.startswith("i"):
            return "g" + _place(init, rhyme, tone)
        return "gi" + _place(init, rhyme, tone)
    if init == "q":
        return "q" + _place(init, rhyme, tone)
    return init + _place(init, rhyme, tone)


def roundtrip_ok(tok):
    p = parse(tok)
    return p is not None and surface(*p) == unicodedata.normalize("NFC", tok)
