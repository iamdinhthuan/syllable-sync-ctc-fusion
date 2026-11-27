# Alignment-Marginalized Multi-Stream Sequence Estimation With CTC for Tonal Speech Recognition

Code and reproducibility package for the manuscript *Alignment-Marginalized Multi-Stream Sequence Estimation With Connectionist Temporal Classification for Tonal Speech Recognition* (under review). Archived on Zenodo: https://doi.org/10.5281/zenodo.24521562 (all versions); v1.0.0 is https://doi.org/10.5281/zenodo.24521563, v1.0.1 adds the dataset download script, and v1.0.2 updates the manuscript title in the documentation.

A Vietnamese syllable is
factorized into initial, rhyme and tone labels; each syllable candidate in beam search advances the
independently marginalized CTC prefixes of all three components, and the evidence is fused with attention and a
syllable language model (LM). Termination uses exact completed-sequence likelihoods.

## Layout

| Path | Content |
|---|---|
| `src/` | Frozen decoding and training code used for every reference-independent result in the paper: IRT tokenizer (`tok.py`), component CTC prefix scorer (`ctc_prefix.py`), syllable-synchronous search with LM proposals and online component fusion (`decode_lm.py`), completed-score rescoring (`nbest_rescore.py`, `rescore_saved_nbest.py`), acoustic model training (`train_vivos.py`), syllable LM (`lm_model.py`), search bounds (`decoding_protocol.py`), replay driver (`run_replay.py`, `replay_manifest.py`), historical character baseline (`train_char.py`, `char_lex_decode.py`), and the frozen replay settings and FLEURS precommit (`*.json`). |
| `retune/` | Common development-retune runner for the component subsets (`run_retune.py`), its frozen plan (`PLAN.json`) and the recorded amendment (`AMENDMENT_1.json`). |
| `bank32/` | The wider completed-bank control (Bank32) reported in the supplement. |
| `lm/` | External text preparation and syllable/character LM training. |
| `data_prep/` | Download of all corpora at pinned revisions (`download_data.py`), conversion of the LSVSC and ViMD Parquet releases, and the character inventory. |
| `tests/` | Exhaustive small-alphabet check of the component CTC prefix recurrences (Supplementary Section S1). |
| `reproducibility/` | Per-utterance records and scripts that regenerate every reference-independent number and data figure of the paper; see `reproducibility/README.md`. |

The files in `src/` are byte-identical to the frozen deployment whose SHA-256 hashes were recorded before
decoding (`SHA256SUMS` lists every file in this repository). Default paths in the scripts (`/data/...`,
`/mnt/ssd2/...`) are those of the original runs; pass the corresponding command-line arguments or edit the
constants to run elsewhere.

## Quick checks (CPU)

```bash
pip install -r requirements.txt
cd reproducibility && python reproduce.py --check      # 319 printed values, ends with CHECK PASSED
python make_figures.py                                  # length, component and retune-curve figures
cd ../tests && CTC_PREFIX_SOURCE=../src/ctc_prefix.py python -B test_ctc_prefix_exhaustive.py -v
```

Training and decoding need a GPU, the speech corpora and the pretrained encoder `facebook/wav2vec2-xls-r-300m`.

## Data

No audio, model weights or ViVOS/ViMD transcripts are included. `data_prep/download_data.py` downloads every
corpus below from the Hugging Face Hub at pinned revisions into the layout the other scripts expect
(`python data_prep/download_data.py --root DATA_DIR`, or `--only vivos fleurs`, or `--dry-run`). FLEURS is pinned to
the exact revision used in the paper and its SHA-256 is verified; the other pins are the revisions at release time.
The corpora keep their own licenses:

- ViVOS: https://huggingface.co/datasets/AILAB-VNUHCM/vivos (original release: https://zenodo.org/records/7068130; CC BY-NC-SA 4.0)
- LSVSC: https://huggingface.co/datasets/doof-ferb/LSVSC (CC BY 4.0)
- ViMD: https://huggingface.co/datasets/nguyendv02/ViMD_Dataset (CC BY-NC-ND 4.0)
- FLEURS (Vietnamese test): https://huggingface.co/datasets/google/fleurs
- LM text: https://huggingface.co/datasets/wikimedia/wikipedia (`20231101.vi`)

The reproducibility records contain normalized references and outputs only for LSVSC and FLEURS (CC BY 4.0);
see `reproducibility/DATA_LICENSES.md`.

## License

Code: MIT (see `LICENSE`). Data records follow the licenses of the source corpora.
