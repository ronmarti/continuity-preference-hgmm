# Continuity Preference in Hidden Gauss-Markov Models
Code for research paper about time-variant linear dynamic model learning by continuity preference. Time-varying Kalman filter trained smartly.

## VS Code devcontainer

1. Install Docker with Docker Compose support and the VS Code **Dev Containers** extension.
2. Open the repository in VS Code and run **Dev Containers: Reopen in Container**.

The configuration in [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)
reuses [docker-compose-dev.yaml](docker-compose-dev.yaml) and the `dev` target in
[Dockerfile](Dockerfile). It mounts the repository at `/app`, retains the existing
network and keep-alive command, and runs as root like the original development
container. Closing the devcontainer stops the Compose service.

VS Code and its terminals use the image's Poetry environment through
`/opt/cp-hgmm-venv`; `PYTHONPATH` includes `/app/cp-hgmm`, and terminals start in
that directory. Python, Pylance, and Jupyter extensions are installed automatically.
No host virtual environment or `.env` file is needed for this workflow.
Run **Dev Containers: Rebuild Container** after changing the Poetry dependencies
or Dockerfile.

The development image includes `curl` and CA certificates so VS Code can download
its CLI during session startup. If an older image fails with `curl: not found`,
run **Dev Containers: Rebuild Container** before starting another session. If the
Agents window cannot rebuild it, run `docker compose -f docker-compose-dev.yaml
build continuity-preference-hgmm` from the repository root, then retry.

The image includes CUDA, but this configuration does not request GPU access,
matching the existing development Compose file. GPU access requires host NVIDIA
Container Toolkit support and an explicit GPU reservation in Compose.

## Legacy local environment

Requires python 3.6 - 3.8 because of the `sktime` dependency for datasets loading and downloading directly from the UCR/UEA database. When `sktime` migrates to higher version of python, this app can migrate as well.
Moreover for the `timeseries_classifier_knn.py`, `numpy` of version `1.20` is required by `sktime`'s `numba` dependency.

Don't mess up your environment, use virtual environment. Useful commands:

`py -3.8 -m venv .venv`
`.venv/Scripts/activate`
`pip install -r requirements.txt`

The root directory for running module is `cp-hgmm`. I suggest to add this directory in your environment variable.

Hint for VSCode: create `.env` in the repository root, inside it, put:
`PYTHONPATH=./cp-hgmm`

## Getting started
* For regression models, try running `cp-hgmm/models/lit_system_em.py`.
* For classification, try running `cp-hgmm/classifiers/timeseries_classifier_cphgmm.py`.
* For massive experiments on the effect of `kappa`, try running `cp-hgmm/experiments/experiment_pareto.py`.
  * For visualization of the experiments results, run `cp-hgmm/experiments/visualization_pareto.ipynb` jupyter notebook.

Try different settings of `kappa` parameter in Continuity-preferring models.
