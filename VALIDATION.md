# Public-release validation

Validated on 2026-11-14 with Python 3.12, NumPy 2.3.5 and PyTorch 2.10.0+cpu.

- All 10 unit tests passed.
- The complete CPU demonstration completed successfully from outside the repository directory.
- Source readout numerical rank: 64; query feature shape: 24 × 40; output shape: 24 × 2.
- Source model weights remained unchanged during target inference.
- Maximum prediction difference among selection rules at the full-library endpoint: 0.
- The fresh synthetic training losses and attribute AP values matched the archived reference run. Elapsed time is machine-dependent.

The release preserves the existing implementation, configuration and generated sample; publication changes concern documentation and packaging. The archived `outputs/root_validation/` files support the worked example in `METHOD_MAP.md`. Regenerate the workflow with `python run_demo.py` and execute the checks with `python -m unittest discover -s tests -v`.

These checks verify the synthetic demonstration only. They do not validate a clinical model or reproduce the formal real-data experiments.
