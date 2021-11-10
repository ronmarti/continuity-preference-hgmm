# continuity-preference-hgmm
Code for research paper about time-variant linear dynamic model learning by continuity preference.

Requires python 3.6 - 3.8 because of the `sktime` dependency for datasets loading and downloading directly from the UCR/UEA database. When `sktime` migrates to higher version of python, this app can migrate as well.
Moreover for the `timeseries_classifier_knn.py`, `numpy` of version `1.20` is required by `sktime`'s `numba` dependency.

Don't mess up your environment, use virtual environment. Useful commands:

`py -3.8 -m venv .venv`
`.venv/Scripts/activate`
`pip install -r requirements.txt`

The root directory for running module is `cp-hgmm`. I suggest to add this directory in your environment variable.

Hint for VSCode: create `.env` in the repository root, inside it, put:
`PYTHONPATH=./cp-hgmm`

# Getting started
For regression models, try running `cp-hgmm/models/lit_system_em.py`.
For classification, try running `cp-hgmm/classifiers/timeseries_classifier_cphgmm.py`.
For massive experiments on the effect of `kappa`, try running `cp-hgmm/experiments/experiment_pareto.py`.
For visualization of the experiments results, run `cp-hgmm/experiments/visualization_pareto.ipynb` jupyter notebook.

Try different settings of `kappa` parameter in Continuity-preferring models.