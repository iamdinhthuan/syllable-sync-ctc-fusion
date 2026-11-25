# Data sources and redistribution

The speech corpora are third-party resources. This package redistributes no audio.

| Corpus | Source | Declared licence | In this package |
|---|---|---|---|
| ViVOS | https://huggingface.co/datasets/AILAB-VNUHCM/vivos | CC BY-NC-SA 4.0 | utterance IDs and counts only; no text |
| LSVSC | https://huggingface.co/datasets/doof-ferb/LSVSC (mirror) | CC BY 4.0 | IDs, counts, decoder outputs, normalized references |
| ViMD | https://huggingface.co/datasets/nguyendv02/ViMD_Dataset | CC BY-NC-ND 4.0 | utterance IDs and counts only; no text |
| FLEURS (vi) | https://huggingface.co/datasets/google/fleurs | CC BY 4.0 | IDs, counts, decoder outputs, normalized references |

Licences are those declared on the repository cards checked on 17 November 2026; they are source
declarations, not an independent rights audit. Normalized references are lowercased,
NFC-normalized, punctuation-free, with underscores replaced by spaces.

## Utterance identifiers

- ViVOS: the official test utterance ID (for example `VIVOSDEV01_R002`); the part before the
  underscore is the speaker.
- ViMD: `<speakerID>_<audio file stem>` of the test split, with underscores inside either part
  replaced by hyphens.
- LSVSC: local IDs `teSSBBB_SSKKKKK`, where `SS` is the index of the test Parquet shard in file
  order, `KKKKK` the row index within that shard and `BBB = KKKKK // 100` the block of up to 100
  consecutive utterances used as resampling unit. They are not speaker labels.
- FLEURS: local IDs in test-file order; references are included.
