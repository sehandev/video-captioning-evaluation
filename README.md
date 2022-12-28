# Evaluation for Video Captioning

Evaluate coherence for video captioning


## TODO
- [x]  Calculate coherence between captions by calculating nsp score (v0.4.0)
- [x]  Load model (v0.3.0)
- [x]  Preprocess dataset (v0.2.0)
- [x]  Init project (v0.1.0)

# How to

## Install dependencies

```bash
$ poetry install
```

## Caption & subtitle conversion

```bash
$ poetry run python evaluation.py \
    --data_dir <path>/data/
```
