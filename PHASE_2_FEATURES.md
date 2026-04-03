
## Blocked: TensorFlow models (tabular_nn, cnn_1d, mc_dropout, deep_ensemble)
Python Python 3.14.3 is not supported by TensorFlow (max: 3.12).
These models are skipped via --skip-nn until a Python 3.12 venv is created.
Planned resolution: create .venv312 with Python 3.12 for TF-dependent training.
