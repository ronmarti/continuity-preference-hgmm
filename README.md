# Continuity Preference in Hidden Gauss-Markov Models
Code for research paper about time-variant linear dynamic model learning by continuity preference. Time-varying Kalman filter trained smartly.

## Original paper and citation

This repository implements the algorithm described in the authors' original paper:

> Martin Ron, Pavel Burget, Václav Hlaváč,
> **Parameter continuity in time-varying Gauss–Markov models for learning from small training data sets**,
> *Information Sciences*, Volume 595, 2022, Pages 197–216, ISSN 0020-0255,
> [https://doi.org/10.1016/j.ins.2022.02.037](https://doi.org/10.1016/j.ins.2022.02.037)
> ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0020025522001724)).

<details>
<summary>Abstract</summary>

The Linear time-invariant dynamic models are widely adopted in the industry. In the
machine learning domain, such models are known as time-invariant continuous-state hidden
Gauss–Markov models. Their super-class, the linear time-varying dynamic models, have
relatively sparse applications as predictive models and classifiers of time series. This
is typically due to the model complexity and the need for a significantly larger training
set than time-invariant models. Without a large training set, a better modeling
performance is counteracted by a less robust model. In this paper, we propose the
continuity preference of the time-varying parameters of the model, which significantly
reduces the required amount of training data while maintaining the modeling performance.
We also derive a simple modification of the Expectation–Maximization algorithm
incorporating continuity in parameters. The modified algorithm shows robust learning
performance. The model performance is demonstrated by experiments on real 6-axis robotic
manipulators in a laboratory, the Skoda Auto car producer body shop, and also on a public
benchmark data set.

*Keywords:* Gauss–Markov process; EM optimization; LTV dynamic models; Kalman filter; Consistency
</details>

If you use this code or the continuity-preference method in your work, please cite the paper:

```bibtex
@article{Ron2022ParameterContinuity,
  title   = {Parameter continuity in time-varying {Gauss--Markov} models for learning from small training data sets},
  author  = {Ron, Martin and Burget, Pavel and Hlav{\'a}{\v{c}}, V{\'a}clav},
  journal = {Information Sciences},
  volume  = {595},
  pages   = {197--216},
  year    = {2022},
  issn    = {0020-0255},
  doi     = {10.1016/j.ins.2022.02.037},
  url     = {https://www.sciencedirect.com/science/article/pii/S0020025522001724}
}
```

## VS Code devcontainer

1. Install Docker with Docker Compose support and the VS Code **Dev Containers** extension.
2. Open the repository in VS Code and run **Dev Containers: Reopen in Container**.

The configuration in [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)
reuses [docker-compose-dev.yaml](docker-compose-dev.yaml) and the `dev` target in
[Dockerfile](Dockerfile). It mounts the repository at `/cp-gmm-app`, retains the existing
network and keep-alive command, and runs as root like the original development
container. Closing the devcontainer stops the Compose service.

VS Code and its terminals use the image's Poetry environment through
`/opt/cp-hgmm-venv`; `PYTHONPATH` includes `/cp-gmm-app/cp-hgmm`, and terminals start in
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

## Web GUI (Solara)

A [Solara](https://solara.dev) web app in [cp-hgmm/gui](cp-hgmm/gui) lets you set up,
start, stop and monitor CP-HGMM training and inspect the results:

* **Setup**: choose a UCR/UEA archive dataset or an uploaded dataset, number of
  hidden states, `kappa`, max epochs, batch size, early-stopping patience and seed.
  Setup stays in the page and stacks above the graphs on narrow screens.
* **Monitor**: live status, epoch counter, loss and continuity-penalty curves, log.
* **Results**: one-step-ahead predictions against test data per sample, per-class
  log predictive density, and a separate parameter continuity
  `‖A_t − A_{t+1}‖_F` plot and best checkpoint for **each class-specific model**.
  Bigger (higher) mean log predictive density is better, including when negative.
* **Run history**: click a run in the history table to inspect its loss and result
  graphs; click it again or use **Show live run** to return to the current job. Confirm
  permanent deletion, then use **Delete selected run** or **Delete all runs**.
  Deletion removes the run's checkpoints and graphs, not uploaded datasets.
  The active run is protected; deleting all runs is disabled during training.

Training runs in a background thread shared by the whole server: one run at a
time, visible from every browser tab, and it keeps going when the page is
closed. Each run writes checkpoints and `summary.json` to
`$CPHGMM_RUNS_DIR/<run_id>` (default `cp-hgmm/runs` when launched from `cp-hgmm`).
New runs also save their evaluation arrays, so graphs remain available after a
server restart without re-training or downloading the dataset again. Older
summary-only runs can show loss history but cannot recover missing result graphs.
Training fits one independent model per training class and evaluates each test
sample with the model for its labeled class; these are predictive-density
diagnostics, not classification accuracy. The supplied test split is also used
for early stopping, so these metrics are not an untouched holdout estimate.
If TEST omits a training class, that model uses its training samples for
early-stopping validation and contributes no samples to the test metric.
UCR datasets are downloaded on first use, so internet access is needed.

### Uploading your own dataset

Select **Upload a dataset** in Setup, enter a unique display name, drop a TRAIN
file and a TEST file, and choose **Validate and save dataset**. It is then selected
under **Dataset source → Uploaded dataset** and remains available in the dropdown
after a restart. **Refresh uploaded datasets** picks up uploads from other tabs.
Archive datasets remain available under **UCR / UEA archive**, including a free
text field for dataset names beyond the suggested list.

Uploads use the UCR/UEA **`.ts` classification format**, not CSV, a ZIP archive or
Python objects. Each file has a metadata header followed by `@data`, with one
sample per line. For example, a small univariate file with labels `a` and `b`:

```text
@problemName MyDataset
@timestamps false
@univariate true
@equalLength true
@classLabel true a b
@data
1.0,2.0,3.0,4.0:a
4.0,3.0,2.0,1.0:b
```

Provide separate TRAIN and TEST files with the same format. For multivariate
series use `@univariate false` and separate channels with colons, followed by
the class label: `1,2,3,4:5,6,7,8:a`. All channels and samples in both files
must share the same series length (at least **3 observations**) and channel count.
The header must declare `@timestamps false`, `@equalLength true`, and
`@classLabel true` with the class names. Each file must be at most **20 MiB**.
Values must be finite real
numbers: missing values (`?`/NaN), infinities, timestamps and unequal-length series
are not supported. Both splits must be nonempty and labeled, and TEST labels must
occur in TRAIN; TEST may omit a training class. Invalid uploads are rejected
with an error before they become selectable. Display names must be unique
(case-insensitively), printable, at most 128 characters and contain no path
separators.

Uploads are stored in `$CPHGMM_DATASETS_DIR`, defaulting to `datasets` beside the
runs directory (`cp-hgmm/datasets` locally, `/data/datasets` in Docker). The
existing Docker `/data` volume persists both runs and uploaded datasets. Do not
expose this shared training service to untrusted users without access control.

### Docker

```sh
docker compose up --build -d
```

Then open http://localhost:8765. The `prod` target of the [Dockerfile](Dockerfile)
starts the GUI on port 8765 and stores runs in the `data` volume (`/data/runs`).
Without Compose:

```sh
docker build --target prod -t continuity-preference-hgmm-gui .
docker run -p 8765:8765 -v cp-hgmm-data:/data continuity-preference-hgmm-gui
```

### Locally or in the devcontainer

From `cp-hgmm` (port 8765 is forwarded by the devcontainer):

```sh
solara run gui/app.py --port 8765
```

In VS Code you can instead start the **Web GUI (Solara server)** configuration from the Run and Debug view (`.vscode/launch.json`), which runs the same server under the debugger.

GUI regression tests (no dataset downloads), from the repository root:

```sh
PYTHONPATH=cp-hgmm python -m unittest discover -s cp-hgmm/gui -p 'test_*.py'
```

## Local environment (without Docker)

Requires Python 3.11 – 3.13 (see `python` in [cp-hgmm/pyproject.toml](cp-hgmm/pyproject.toml)).
Dependencies are managed by [Poetry](https://python-poetry.org/), which creates an
isolated virtual environment for you. From the `cp-hgmm` directory:

```sh
poetry install --no-root
eval "$(poetry env activate)"   # or prefix commands with `poetry run`
```

The root directory for running modules is `cp-hgmm`; add it to `PYTHONPATH`.

Hint for VS Code: create `.env` in the repository root, inside it, put:
`PYTHONPATH=./cp-hgmm`

`cp-hgmm/classifiers/timeseries_classifier_knn.py` uses `sktime`'s
`KNeighborsTimeSeriesClassifier`, which needs the optional `numba` dependency
(`poetry run pip install numba`).

The pinned [requirements.txt](requirements.txt) is a legacy snapshot of the original
Python 3.8 environment and is not maintained; prefer the Poetry setup above.

## Try it out
* For regression models, try running `cp-hgmm/models/lit_system_em.py`.
* For classification, try running `cp-hgmm/classifiers/timeseries_classifier_cphgmm.py`.
* For massive experiments on the effect of `kappa`, try running `cp-hgmm/experiments/experiment_pareto.py`.
  * For visualization of the experiments results, run `cp-hgmm/experiments/visualization_pareto.ipynb` jupyter notebook.

Try different settings of `kappa` parameter in Continuity-preferring models.
