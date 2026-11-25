# Reproducibility package

Supplementary data for "Syllable-Synchronous Component CTC Fusion for Tonal Speech Recognition:
A Vietnamese Case Study".

The package contains the saved decoder outputs behind the reference-independent result tables and
data figures of the article and its supplement, as per-utterance records, together with the scripts that regenerate the reported numbers
and figures from them. It needs no GPU, audio or model weights.

## Quick start

```bash
python -m pip install -r requirements.txt     # numpy, matplotlib
python reproduce.py --check                   # recompute all reported numbers and compare
python make_figures.py                        # regenerate the data figures into out/
python show_examples.py                       # output cases selected by the fixed rule (--corpus FLEURS)
python verify_edits.py                        # recount edits from text (LSVSC and FLEURS)
```

`reproduce.py --check` ends with `CHECK PASSED` when each recomputed value equals the value
printed in the article or its supplement at exactly the printed number of decimals
(`expected_values.json`).

## What regenerates what

| Article item | Records | Script |
|---|---|---|
| Table V, main results (greedy, LM, rescoring, rescue, Full8, caps) | `data/<corpus>/main_*.json` | `reproduce.py` |
| Sec. IV-C, rescoring-minus-Full8 intervals on covered utterances | `main_rescore.json`, `main_full8.json` | `reproduce.py` (`paired_numpy`) |
| Sec. IV-C, ViVOS without capped utterances; 566 nonrecurring sentences | `main_lm.json`, `train_text_recurrence_uids.json` | `reproduce.py` |
| Table VI, component retune and subset-minus-IRT intervals | `component_retuned_*.json`, `component_retune_settings.json` | `reproduce.py` (`paired_python`) |
| Fig. 2, WER by reference length | `main_lm/rescue/full8.json` | `make_figures.py` |
| Fig. 3, output cases (selection rule) | LSVSC `main_lm.json`, `main_full8.json` | `show_examples.py` |
| Fig. 4, component subsets | `component_retuned_*.json`, `component_fixed_*.json` | `make_figures.py` |
| Supplement, pool-oracle table | `component_fixed_*.json` (field `oe`) | `reproduce.py` |
| Supplement, Bank32 tables and length figure | `main_bank32.json` | `reproduce.py`, `make_figures.py` |
| Supplement, retune development curves | `component_retune_settings.json` | `make_figures.py` |
| Supplement, additional output cases | LSVSC records | `show_examples.py --latex` |
| Table IV, decoding weights; Table VII, timing | `decoding_weights.json`, `timing.json` | values as stored |
| Sec. IV-A and supplement Section S9, training trials (development CER, best epoch) | `training_trials.json` | `reproduce.py` |

Fig. 1 (architecture) is a schematic and has no data.

## Record format

Each `data/<corpus>/<system>.json` holds `{"corpus", "system", "utterances": {uid: record}}` with

- `n`, `e`: reference token count and token edit distance (WER = sum e / sum n);
- `nc`, `ec`: reference character count and character edit distance (CER);
- `hyp`, `ref`: the decoder output and the normalized reference (LSVSC and FLEURS only);
- `text_withheld`, `has_digit`: one LSVSC utterance (`te01000_0100034`) carries counts only, because its
  transcript states a named individual's date and place of birth; `has_digit` keeps the case-selection
  rule computable;
- `cap`: output reached the acoustic search bound (LM and Full8);
- `empty_pool`: rescoring had no finite completed candidate (rescoring);
- `oe`: minimum token edits over the saved candidate pool (`null` if the pool is empty).

The resampling unit of an utterance is the part of its ID before the first underscore: a ViVOS
speaker, a block of up to 100 consecutive LSVSC utterances, or a ViMD supplied speaker ID.

Systems: `main_greedy` (unconstrained attention decoding), `main_lm` (development-retuned LM
proposals), `main_rescore` (completed-score rescoring), `main_rescue` (rescoring with greedy output
for empty pools), `main_full8` (online component fusion, beam 8), `main_bank32`,
`component_fixed_<S>` (subset S at the Full8 fusion settings) and `component_retuned_<S>` (subset S
after the common development retune), with S in I, R, T, IR, IRT.

## Text of references and outputs

See `DATA_LICENSES.md`. Normalized references and decoder outputs are included for LSVSC and FLEURS
(CC BY 4.0). ViVOS (CC BY-NC-SA) and ViMD (CC BY-NC-ND) restrict redistribution, and decoder outputs
are near-copies of their transcripts, so for these two corpora the package carries only the
per-utterance counts (`n`, `e`, `nc`, `ec`) and flags. These counts are sufficient for every WER, CER,
interval, length profile and pool oracle that `reproduce.py` regenerates.

## Scope and limits

- Not covered: the historical three-seed and character tables, the reduced-data figure, the legacy
  fixed-bonus replay and the lexical-coverage numbers of supplement Section S6 have no records here.
  Development outputs are not included, except the development WER/CER of every component-retune
  setting and the logged per-epoch development CER of the training runs in `training_trials.json`
  (greedy decoding, as recorded during training; the outputs behind them are not included).
- This package contains no decoder or training code; the scripts only analyse saved outputs. The decoding and training code is in `../src` of the code repository.
- Apart from the training trials, all numbers are seed-zero, reference-independent decoding results
  of one recovered checkpoint per corpus. Acoustic training and decoding cannot be repeated from this package.
- The two families of bootstrap intervals were produced by two generators (NumPy for the rescoring
  comparison, Python `random` for the component table), both with seed 20261117 and 10,000
  group resamples; the scripts keep each as originally run.
- Candidate pools (n-best lists) are not included; their per-utterance oracle edits are.
- Timing values are stored as measured on one H100 session and are not regenerated.
