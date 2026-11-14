# LGCS: Lesion-Graph Context Selection Demo

Runnable companion demonstration for **Lesion-Graph Representations and Context Selection for Label-Efficient Multi-Disease Fundus Screening**.

[中文说明](README.zh-CN.md) · [Method-to-code map](METHOD_MAP.md) · [Data card](data/DATA_CARD.md) · [Data and code availability](AVAILABILITY.md)

**Release scope:** this is a small, synthetic-data demonstration. It trains a tiny graph encoder and uses an explicitly labelled deterministic RBF context predictor. It is not a reproduction of official SIGraph/TabICL models, a complete implementation of the manuscript experiments, or evidence of clinical performance. No real retinal images, patient records, or third-party pretrained weights are included or downloaded.

## Quick start

Tested with Python 3.12, NumPy 2.3.5 and CPU PyTorch 2.10.0. Create and activate a Python environment, then run:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5
python run_demo.py
python -m unittest discover -s tests -v
```

The entry point resolves its default configuration and data relative to its own directory. Use `--output PATH` to choose a results directory and `--config PATH` to supply a configuration. No GPU or external model download is required.

## What runs

1. Train a randomly initialized tiny CNN/graph model on synthetic source grades, then freeze its weights.
2. Fit a 32-component PCA and retrieval scales using source rows only; append eight hard-assignment spatial measurements to obtain 40 columns.
3. For each of two synthetic binary attributes, build one labelled library of 20 positive and 20 negative examples (`B=40`). Select 5 positive and 5 negative support rows (`k=10`) using graph coverage, random or cosine selection.
4. Fit predictor preprocessing on the selected support rows only. Predict query attributes and evaluate grouped-tie average precision after prediction.
5. Check frozen source weights and identical predictions for all selection rules at the full-library endpoint (`k=B=40`).

The 40 revealed labels per attribute remain the acquisition budget even when only 10 rows enter a context. Query labels are not passed to the selector or predictor.

## Data and outputs

`data/sample.npz` contains 96 source, 80 pool and 24 query geometric phantoms at 32×32 pixels. The labels describe generated spot-count and radial attributes, not clinical diagnoses. The arrays contain no real patient information. Regenerate them with:

```bash
python data/generate_data.py
```

The seed is 20261030. Compare array values for reproducibility; an NPZ file's container metadata may affect its byte hash. Existing metadata records the distributed sample's SHA-256. The generated arrays carry the existing CC0-1.0 designation documented in the data card; this designation does not apply to external datasets, dependencies, or code.

`outputs/root_validation/` contains the earlier synthetic reference run used by the worked example in `METHOD_MAP.md`. A new run writes to `outputs/demo/`: synthetic model weights, source PCA/scales, predictions, graph measurements, label/context ledgers, configuration and metrics. The included model was trained only on the generated phantoms. These outputs must not be reported as real-dataset results. Timing depends on the machine.

## Implementation boundaries

The compact encoder substitutes for the formal graph backbone. The default RBF learner substitutes for TabICL; the `ContextPredictor(estimator_factory=...)` interface is an extension point, while `run_demo.py` deliberately rejects an unsupported backend name. The formal model integration, real-data patient splits and disease dictionary, global representation controls, external baselines, 30 support draws and patient bootstrap experiments remain outside this demo. See `METHOD_MAP.md` for module-level detail.

For future experiments, obtain EyePACS and RFMiD from their original providers under their applicable access conditions. Provider links and the publication statement are in `AVAILABILITY.md`. This repository does not redistribute either dataset.

## Repository contents

| Path | Purpose |
|---|---|
| `run_demo.py`, `config.json` | CPU execution and fixed demonstration defaults |
| `src/core.py` | Tiny graph model, spatial measurements, source interface, selectors, predictor and AP |
| `data/` | Synthetic sample, generator, metadata and data card |
| `tests/test_core.py` | Ten interface and boundary checks |
| `outputs/root_validation/` | Archived synthetic reference outputs |
| `VALIDATION.md` | Public-release verification record |

This repository is a research demonstration and has not been validated for clinical use.
