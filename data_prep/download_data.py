"""Download the speech corpora and LM text used in the paper from the Hugging Face Hub.

    pip install huggingface_hub
    python data_prep/download_data.py --root /path/to/vispeech_data            # everything (~100 GB)
    python data_prep/download_data.py --root /path/to/vispeech_data --only vivos fleurs
    python data_prep/download_data.py --dry-run                                # list files at the pinned revisions

Layout written under --root (the layout the other scripts expect; the original runs used
--root /mnt/ssd2/vispeech_data, which is the default path inside prep_parquet.py and build_lm_corpus.py):

    vivos/vivos/{train,test}/...          ViVOS, extracted from data/vivos.tar.gz
    LSVSC/data/*.parquet                  LSVSC (convert with prep_parquet.py)
    ViMD_Dataset/data/*.parquet           ViMD (convert with prep_parquet.py)
    viwiki/20231101.vi/*.parquet          Vietnamese Wikipedia snapshot for the LM text (build_lm_corpus.py)
    fleurs/vi_vn/test/0000.parquet        FLEURS Vietnamese test split (transfer test)

Revisions are pinned. FLEURS is pinned to the exact revision recorded before the transfer test
(src/EXTERNAL_EVAL_PRECOMMIT.json) and its SHA-256 is checked. The other pins are the repository revisions at
the time of this release; the original runs downloaded the same repositories earlier, before revisions were
recorded. Each corpus keeps its own license (see README.md); ViVOS and ViMD are non-commercial.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

SOURCES = {
    'vivos': dict(repo='AILAB-VNUHCM/vivos', revision='3cbfb2502e5e84776b4b778b020a09759f723f52',
                  patterns=['data/vivos.tar.gz'], dest='vivos'),
    'lsvsc': dict(repo='doof-ferb/LSVSC', revision='269de0604fd94765648cbfcaed4341ef65a31e50',
                  patterns=['data/*.parquet'], dest='LSVSC'),
    'vimd': dict(repo='nguyendv02/ViMD_Dataset', revision='3a5b30157034e7eadd5c75fae1a820c6f9383398',
                 patterns=['data/*.parquet'], dest='ViMD_Dataset'),
    'wiki': dict(repo='wikimedia/wikipedia', revision='b04c8d1ceb2f5cd4588862100d08de323dccfbaa',
                 patterns=['20231101.vi/*.parquet'], dest='viwiki'),
    'fleurs': dict(repo='google/fleurs', revision='168de341b3db6859a9bac1c50a2ef5e3b47647e0',
                   patterns=['vi_vn/test/0000.parquet'], dest='fleurs',
                   sha256='65325be18bef84bb97b5bad1145958ae752919bb0b6696e7cfa8409ef5c46c77'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data', help='output directory (default: ./data)')
    ap.add_argument('--only', nargs='+', choices=sorted(SOURCES), help='download only these sources')
    ap.add_argument('--dry-run', action='store_true', help='list the files at the pinned revisions and exit')
    a = ap.parse_args()
    from fnmatch import fnmatch
    from huggingface_hub import HfApi, snapshot_download

    api, root = HfApi(), Path(a.root)
    for name in a.only or SOURCES:
        s = SOURCES[name]
        files = [f for f in api.list_repo_files(s['repo'], repo_type='dataset', revision=s['revision'])
                 if any(fnmatch(f, p) for p in s['patterns'])]
        assert files, f"{name}: no files matched at revision {s['revision']}"
        print(f"{name}: {s['repo']}@{s['revision'][:10]} -> {root / s['dest']} ({len(files)} files)")
        if a.dry_run:
            continue
        dest = root / s['dest']
        snapshot_download(s['repo'], repo_type='dataset', revision=s['revision'], allow_patterns=s['patterns'],
                          local_dir=dest)
        if 'sha256' in s:
            got = sha256(dest / files[0])
            assert got == s['sha256'], f"{name}: SHA-256 mismatch ({got})"
            print(f"  SHA-256 verified: {got}")
        if name == 'vivos':
            archive = dest / 'data/vivos.tar.gz'
            with tarfile.open(archive) as tar:
                tar.extractall(dest, filter='data')       # creates vivos/vivos/{train,test}
            print(f"  extracted {archive.name} -> {dest / 'vivos'}")
    if not a.dry_run:
        (root / 'DOWNLOAD_REVISIONS.json').write_text(json.dumps(
            {n: dict(repo=SOURCES[n]['repo'], revision=SOURCES[n]['revision']) for n in a.only or SOURCES}, indent=1))


if __name__ == '__main__':
    main()
