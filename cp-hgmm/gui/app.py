"""Solara web GUI for setting up, running, monitoring and inspecting CP-HGMM training.

Run from the `cp-hgmm` directory:  solara run gui/app.py
"""
import sys
import threading
import time
from pathlib import Path

# Make `models` / `gui` importable regardless of how solara was launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ipyvuetify as v  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import solara  # noqa: E402
import traitlets  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from gui.datasets import (  # noqa: E402
    DATASET_FORMAT_HELP, MAX_UPLOAD_BYTES, list_uploaded_datasets, save_uploaded_dataset,
)
from gui.training import (  # noqa: E402
    TrainingConfig, delete_all_runs, delete_run, job, list_runs, load_run,
)

UCR_DATASETS = ['Coffee', 'GunPoint', 'ECG200', 'ItalyPowerDemand', 'Beef', 'Wine', 'TwoLeadECG']
STATUS_COLORS = {
    'idle': 'grey', 'loading': 'info', 'training': 'primary', 'evaluating': 'info',
    'finished': 'success', 'stopped': 'warning', 'failed': 'error',
}

defaults = TrainingConfig()
dataset = solara.reactive(defaults.dataset)
num_states = solara.reactive(defaults.num_states)
kappa = solara.reactive(defaults.kappa)
max_epochs = solara.reactive(defaults.max_epochs)
batch_size = solara.reactive(defaults.batch_size)
patience = solara.reactive(defaults.patience)
seed = solara.reactive(defaults.seed)
dataset_source = solara.reactive('UCR / UEA archive')
uploaded_dataset = solara.reactive('')


@solara.component
def DatasetUpload(on_saved):
    name = solara.use_reactive('')
    train_file = solara.use_reactive(None)
    test_file = solara.use_reactive(None)
    error = solara.use_reactive('')
    upload_generation = solara.use_reactive(0)

    def receive(file, target):
        error.value = ''
        target.value = None
        try:
            if file['size'] > MAX_UPLOAD_BYTES:
                raise ValueError(f'Each file must be at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.')
            content = file['file_obj'].read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise ValueError('File exceeds the upload size limit.')
            target.value = (file['name'], content)
        except Exception as err:
            error.value = f'Upload failed: {err}'

    def save():
        error.value = ''
        try:
            train_name, train_content = train_file.value
            test_name, test_content = test_file.value
            dataset_id = save_uploaded_dataset(
                name.value, train_name, train_content, test_name, test_content,
            )
            train_file.value = None
            test_file.value = None
            name.value = ''
            upload_generation.value += 1
            on_saved(dataset_id)
        except Exception as err:
            error.value = f'Could not save dataset: {err}'

    with solara.Details('Upload a dataset'):
        solara.Markdown(DATASET_FORMAT_HELP)
        solara.InputText('Dataset display name', value=name, continuous_update=True)
        solara.FileDrop('Drop TRAIN .ts file here', on_file=lambda f: receive(f, train_file)).key(
            f'train-{upload_generation.value}')
        if train_file.value:
            solara.Text(f'TRAIN: {train_file.value[0]}')
        solara.FileDrop('Drop TEST .ts file here', on_file=lambda f: receive(f, test_file)).key(
            f'test-{upload_generation.value}')
        if test_file.value:
            solara.Text(f'TEST: {test_file.value[0]}')
        solara.Button(
            'Validate and save dataset', on_click=save,
            disabled=not name.value.strip() or train_file.value is None or test_file.value is None,
        )
        if error.value:
            solara.Error(error.value)


def use_job_version(interval: float = 0.5) -> int:
    """Re-renders the caller whenever the shared training job changes."""
    version = solara.use_reactive(job.version)

    def poll(cancel: threading.Event):
        while not cancel.wait(interval):
            if job.version != version.value:
                version.value = job.version

    solara.use_thread(poll, dependencies=[], intrusive_cancel=False)
    return version.value


@solara.component
def SetupPanel(running: bool, on_start=None):
    error = solara.use_reactive('')
    upload_version = solara.use_reactive(0)

    def read_uploads():
        try:
            return list_uploaded_datasets(), ''
        except Exception as err:
            return [], f'Could not list uploaded datasets: {err}'

    uploads, upload_error = solara.use_memo(read_uploads, dependencies=[upload_version.value])
    upload_names = {r['name']: r['id'] for r in uploads}
    selected_name = next((name for name, key in upload_names.items()
                          if key == uploaded_dataset.value), None)

    def saved(dataset_id):
        uploaded_dataset.value = dataset_id
        dataset_source.value = 'Uploaded dataset'
        upload_version.value += 1

    def start():
        error.value = ''
        try:
            job.start(TrainingConfig(
                dataset=(dataset.value.strip() if dataset_source.value == 'UCR / UEA archive'
                         else uploaded_dataset.value),
                dataset_source=('ucr' if dataset_source.value == 'UCR / UEA archive' else 'uploaded'),
                dataset_name=(dataset.value.strip() if dataset_source.value == 'UCR / UEA archive'
                              else selected_name or ''),
                num_states=int(num_states.value),
                kappa=float(kappa.value), max_epochs=int(max_epochs.value),
                batch_size=int(batch_size.value), patience=int(patience.value),
                seed=int(seed.value),
            ))
            if on_start is not None:
                on_start()
        except Exception as err:
            error.value = str(err)

    solara.Select('Dataset source', values=['UCR / UEA archive', 'Uploaded dataset'],
                  value=dataset_source, disabled=running)
    if dataset_source.value == 'UCR / UEA archive':
        solara.Select('UCR dataset', values=UCR_DATASETS, value=dataset,
                      disabled=running).key('archive-dataset')
        solara.InputText('…or any UCR/UEA dataset name', value=dataset, disabled=running)
        solara.Markdown('Archive datasets are downloaded on first use (internet required).')
    else:
        solara.Select('Uploaded dataset', values=list(upload_names), value=selected_name,
                      on_value=lambda name: setattr(uploaded_dataset, 'value', upload_names.get(name, '')),
                      disabled=running).key('uploaded-dataset')
        solara.Button('Refresh uploaded datasets', on_click=lambda: setattr(
            upload_version, 'value', upload_version.value + 1))
        if not uploads:
            solara.Info('Upload a TRAIN/TEST pair below to get started.')
    if upload_error:
        solara.Error(upload_error)
    if not running:
        DatasetUpload(on_saved=saved)
    with solara.Div(style={'margin-top': '16px'}):
        solara.SliderInt(f'Hidden states: {num_states.value}', value=num_states, min=1, max=30,
                         thumb_label=False, tick_labels='end_points',
                         disabled=running).key('hidden-states')
    solara.InputFloat('κ (continuity precision)', value=kappa, disabled=running).key('kappa')
    solara.Markdown(
        '<small>Higher κ ⇒ smoother time-varying A<sub>t</sub>. '
        '0 = plain time-varying Kalman, negative = time-invariant.</small>'
    )
    solara.InputInt('Max epochs', value=max_epochs, disabled=running).key('max-epochs')
    solara.InputInt('Batch size', value=batch_size, disabled=running).key('batch-size')
    solara.InputInt('Early-stopping patience', value=patience, disabled=running).key('patience')
    solara.InputInt('Random seed', value=seed, disabled=running).key('seed')
    with solara.Row(style={'flex-wrap': 'wrap'}):
        solara.Button('Start training', icon_name='mdi-play', color='primary',
                      on_click=start, disabled=running)
        solara.Button('Stop', icon_name='mdi-stop', color='error',
                      on_click=job.stop, disabled=not running)
    if error.value:
        solara.Error(error.value)


def _loss_figure(history):
    fig = Figure(figsize=(9, 3.6), layout='constrained')
    ax_loss, ax_reg = fig.subplots(1, 2)
    df = pd.DataFrame(history)
    groups = df.groupby('class', sort=False) if 'class' in df else [(None, df)]
    for label, rows in groups:
        prefix = f'Class {label}: ' if label is not None else ''
        for name, style in (('aug_loss', '-'), ('val_aug_loss', '--'), ('nll', '-'), ('val_nll', '--')):
            if name in rows:
                ax_loss.plot(rows['epoch'], rows[name], style, marker='.', label=prefix + name)
        if 'mean_frob_norm_sum' in rows:
            ax_reg.plot(rows['epoch'], rows['mean_frob_norm_sum'], marker='.',
                        label=f'Class {label}' if label is not None else 'model')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_title('Loss')
    ax_loss.legend(fontsize='small')
    ax_loss.grid(alpha=0.3)
    if ax_reg.lines:
        ax_reg.legend(fontsize='small')
    ax_reg.set_xlabel('epoch')
    ax_reg.set_title('Continuity penalty (mean κ·‖ΔA‖)')
    ax_reg.grid(alpha=0.3)
    for ax in (ax_loss, ax_reg):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


@solara.component
def MonitorPanel(snap):
    active = snap.status in ('loading', 'training', 'evaluating')
    end = snap.finished if snap.finished is not None else (time.time() if active else None)
    elapsed = end - snap.started if end is not None and snap.started is not None else None
    loss_fig = solara.use_memo(lambda: _loss_figure(snap.history) if snap.history else None,
                               dependencies=[snap.run_id, snap.version])
    with solara.Card('Training monitor'):
        with solara.Row(style={'align-items': 'center', 'flex-wrap': 'wrap'}):
            solara.v.Chip(children=[snap.status.upper()], color=STATUS_COLORS.get(snap.status, 'grey'),
                          text_color='white')
            if snap.run_id:
                solara.Text(f'Run {snap.run_id}')
                if any('class' in row for row in snap.history):
                    solara.Text(f'Completed epochs: {len(snap.history)} '
                                f'(limit {snap.config.max_epochs} per class)')
                else:
                    solara.Text(f'Epoch {snap.epoch} / {snap.config.max_epochs}')
                if getattr(snap, 'current_class', None) is not None:
                    solara.Text(f'Class {snap.current_class}')
                if elapsed is not None:
                    solara.Text(f'Elapsed {elapsed:.1f} s')
                val = [r['val_aug_loss'] for r in snap.history if 'val_aug_loss' in r]
                if val:
                    solara.Text(f'Best val_aug_loss {min(val):.4f}')
        if snap.status in ('loading', 'training', 'evaluating'):
            solara.ProgressLinear(True)
        if snap.error:
            solara.Error(snap.error)
        if loss_fig is not None:
            solara.FigureMatplotlib(loss_fig, dependencies=[snap.run_id, snap.version], format='png')
        with solara.Details('Log', expand=snap.status == 'failed'):
            solara.Preformatted('\n'.join(snap.log[-200:]) or '(empty)')


def _sample_figure(result, idx):
    y, yp = result.y_test[idx], result.y_pred[idx]
    fig = Figure(figsize=(9, 3.2), layout='constrained')
    ax = fig.subplots()
    t = np.arange(y.shape[0])
    for d in range(y.shape[1]):
        ax.plot(t, y[:, d], color=f'C{d}', label=f'data[{d}]')
        ax.plot(t, yp[:, d], '--', color=f'C{d}', alpha=0.8, label=f'prediction[{d}]')
    ax.set_title(f'Test sample #{idx} (class {result.labels[idx]}, '
                 f'mean log pred. density {result.sample_lpd[idx]:.3f})')
    ax.set_xlabel('t')
    ax.legend(fontsize='small', ncol=2)
    ax.grid(alpha=0.3)
    return fig


def _model_figure(result):
    fig = Figure(figsize=(9, 3.2), layout='constrained')
    ax_lpd = fig.subplots()
    classes = list(dict.fromkeys(result.labels))
    ax_lpd.boxplot([result.sample_lpd[result.labels == c] for c in classes])
    ax_lpd.set_xticks(range(1, len(classes) + 1), [str(c) for c in classes])
    ax_lpd.set_xlabel('class')
    ax_lpd.set_title('Mean log predictive density per test sample')
    ax_lpd.set_ylabel('Mean log predictive density (higher is better)')
    ax_lpd.grid(alpha=0.3)
    return fig


def _continuity_figure(label, norms):
    fig = Figure(figsize=(9, 3.2), layout='constrained')
    ax_a = fig.subplots()
    if norms.size:
        ax_a.plot(norms, color='tab:purple')
    else:
        ax_a.text(0.5, 0.5, 'time-invariant model', ha='center', va='center', transform=ax_a.transAxes)
    ax_a.set_xlabel('t')
    ax_a.set_title(f'Class {label}: parameter continuity ‖A$_t$ − A$_{{t+1}}$‖$_F$')
    ax_a.grid(alpha=0.3)
    return fig


@solara.component
def ResultsPanel(run_id: str, result):
    idx = solara.use_reactive(0)
    n = result.y_test.shape[0]
    sample = min(idx.value, n - 1)
    sample_fig = solara.use_memo(lambda: _sample_figure(result, sample), dependencies=[run_id, sample])
    model_fig = solara.use_memo(lambda: _model_figure(result), dependencies=[run_id])
    continuity_figs = solara.use_memo(
        lambda: [(label, _continuity_figure(label, norms))
                 for label, norms in result.a_diff_norms.items()],
        dependencies=[run_id],
    )
    with solara.Card('Results (best checkpoint on test set)'):
        solara.Markdown(f'**Test ELPD (average over all test classes):** {result.test_elpd:.4f}')
        solara.Markdown('<small>Mean log predictive density per time step, averaged over every '
                        'test sample of all classes (each scored by its own class model), so '
                        'larger classes weigh more. See the per-class plot below.</small>')
        if not getattr(result, 'checkpoints', None):
            solara.Text(f'Checkpoint: {result.checkpoint or "n/a"}')
        solara.Markdown('**Bigger (higher) mean log predictive density is better**, '
                        'including for negative values. Test ELPD uses the same convention.')
        solara.SliderInt('Test sample', value=idx, min=0, max=n - 1)
        solara.FigureMatplotlib(sample_fig, dependencies=[run_id, sample], format='png')
        solara.FigureMatplotlib(model_fig, dependencies=[run_id], format='png')
        for label, figure in continuity_figs:
            solara.FigureMatplotlib(figure, dependencies=[run_id, label], format='png')
        if getattr(result, 'checkpoints', None):
            with solara.Details('Checkpoints per class'):
                for label, checkpoint in result.checkpoints.items():
                    solara.Text(f'Class {label}: {checkpoint or "last model state"}')


class RunTableWidget(v.VuetifyTemplate):
    """Single-select run table; clicking a row selects it, clicking it again deselects."""
    template = traitlets.Unicode('''
<template>
  <v-data-table :headers="[...headers]" :items="[...items]" item-value="run_id"
                hover density="compact" :items-per-page="10" class="run-table">
    <template v-slot:headers>
      <tr>
        <th style="width: 40px"></th>
        <th v-for="header in headers" :key="header.key" class="text-no-wrap">{{ header.title }}</th>
      </tr>
    </template>
    <template v-slot:item="props">
      <tr :class="{'run-table__selected': props.item.run_id === selected}" style="cursor: pointer"
          @click="select_run(props.item.run_id === selected ? null : props.item.run_id)">
        <td style="width: 40px">
          <v-icon :color="props.item.run_id === selected ? 'primary' : undefined">
            {{ props.item.run_id === selected ? 'mdi-radiobox-marked' : 'mdi-radiobox-blank' }}
          </v-icon>
        </td>
        <td v-for="header in headers" :key="header.key" class="text-no-wrap">
          {{ props.item[header.key] }}
        </td>
      </tr>
    </template>
  </v-data-table>
</template>
<style>
.run-table .run-table__selected { background: rgba(var(--v-theme-primary), 0.12); }
</style>
''').tag(sync=True)
    headers = traitlets.List().tag(sync=True)
    items = traitlets.List().tag(sync=True)
    selected = traitlets.Unicode(None, allow_none=True).tag(sync=True)

    def vue_select_run(self, run_id):
        self.selected = run_id or None


HISTORY_HEADERS = [
    {'title': 'Run', 'key': 'run_id'},
    {'title': 'Status', 'key': 'status'},
    {'title': 'Dataset', 'key': 'dataset'},
    {'title': 'States', 'key': 'states'},
    {'title': 'κ', 'key': 'kappa'},
    {'title': 'Total epochs', 'key': 'epochs'},
    {'title': 'Best val_aug_loss', 'key': 'best_val_aug_loss'},
    {'title': 'Test ELPD (all classes)', 'key': 'test_elpd'},
]


def _cell(value):
    """JSON-safe, compact table value (widgets cannot transport NaN)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return None if not np.isfinite(number) else round(number, 4)


def _dataset_label(config, uploaded_names):
    """Display name of a run's dataset; uploaded datasets are stored under an opaque id."""
    config = config or {}
    dataset = config.get('dataset', '')
    if config.get('dataset_name'):
        return config['dataset_name']
    if config.get('dataset_source') == 'uploaded':
        return uploaded_names.get(dataset, f'{dataset} (deleted upload)')
    return dataset


@solara.component
def RunTable(runs, selected_run, on_select):
    """Single-select table of saved runs; clicking a row toggles its selection."""
    def read_upload_names():
        try:
            return {d['id']: d['name'] for d in list_uploaded_datasets()}
        except Exception:
            return {}

    uploaded_names = solara.use_memo(read_upload_names, dependencies=[len(runs)])
    rows = [{
        'run_id': r['run_id'], 'status': r['status'],
        'dataset': _dataset_label(r.get('config'), uploaded_names),
        'states': r['config']['num_states'], 'kappa': _cell(r['config']['kappa']),
        'epochs': len(r.get('history', [])) if 'history' in r else r['epochs'],
        'best_val_aug_loss': _cell(r['best_val_aug_loss']),
        'test_elpd': _cell(r['test_elpd']),
    } for r in runs]
    return RunTableWidget.element(
        headers=HISTORY_HEADERS, items=rows, selected=selected_run,
        on_selected=lambda run_id: on_select(run_id or None),
    )


@solara.component
def HistoryPanel(version: int, selected_run, on_select, running: bool, active_run):
    confirmed = solara.use_reactive(False)
    error = solara.use_reactive('')

    def read_runs():
        try:
            return list_runs(), ''
        except Exception as err:
            return [], f'Could not list saved runs: {err}'

    runs, list_error = solara.use_memo(read_runs, dependencies=[version])

    def select(value):
        confirmed.value = False
        error.value = ''
        on_select(value or None)

    def delete(all_runs=False):
        error.value = ''
        try:
            if not confirmed.value:
                raise ValueError('Confirm permanent deletion first.')
            if all_runs:
                delete_all_runs()
            else:
                delete_run(selected_run)
            confirmed.value = False
            on_select(None)
        except Exception as err:
            error.value = f'Could not delete runs: {err}'

    with solara.Card('Run history'):
        if list_error:
            solara.Error(list_error)
        if not runs and not list_error:
            solara.Text('No runs yet.')
        if runs:
            solara.Markdown('<small>Click a run to view its graphs; click it again '
                            '(or use “Show live run”) to return to the live view.</small>')
            with solara.Div(style={'overflow-x': 'auto', 'max-width': '100%'}):
                RunTable(runs, selected_run, select).key('history-table')
        solara.Button('Show live run', icon_name='mdi-play-circle-outline', text=True,
                      on_click=lambda: select(None), disabled=not selected_run)
        solara.Checkbox(label='Confirm permanent deletion of run data and checkpoints',
                        value=confirmed, disabled=not runs)
        with solara.Row(style={'flex-wrap': 'wrap'}):
            solara.Button('Delete selected run', icon_name='mdi-delete', color='error',
                          on_click=lambda: delete(False),
                          disabled=not confirmed.value or not selected_run
                          or (running and selected_run == active_run))
            solara.Button('Delete all runs', icon_name='mdi-delete-sweep', color='error',
                          on_click=lambda: delete(True),
                          disabled=not confirmed.value or not runs or running)
        if running:
            solara.Text('The active run cannot be deleted. Stop training before deleting all runs.')
        if error.value:
            solara.Error(error.value)


def _read_saved_run(run_id):
    if run_id is None:
        return None, ''
    try:
        return load_run(run_id), ''
    except Exception as err:
        return None, f'Could not open run {run_id}: {err}'


@solara.component
def Page():
    version = use_job_version()
    snap = job.snapshot()
    running = job.is_running
    selected_run, set_selected_run = solara.use_state(None)
    saved_snap, saved_error = solara.use_memo(
        lambda: _read_saved_run(selected_run), dependencies=[selected_run, version if not running else -1],
    )
    displayed = saved_snap if selected_run else snap
    solara.Title('Continuity-preferring HGMM — training GUI')
    with solara.AppBar():
        solara.AppBarTitle('Continuity-preferring HGMM — training GUI')
    with solara.Column(classes=['gui-main'], style={'padding': '12px', 'min-width': '0'}):
        solara.Style("""
            .v-application, .v-application__wrap { min-width: 0; width: 100%; }
            .gui-main { min-width: 0; width: 100%; }
            .gui-main .v-row, .gui-main [class*="v-col"] { min-width: 0; }
            .gui-main .v-card, .gui-main .v-input { min-width: 0; max-width: 100%; }
            .gui-main .v-btn { max-width: 100%; height: auto; min-height: 36px; }
            .gui-main .v-btn__content { white-space: normal; }
            .gui-main img { max-width: 100%; height: auto; }
            .gui-main p, .gui-main pre { overflow-wrap: anywhere; }
            .gui-main .v-label { white-space: normal; }
            .v-app-bar .v-toolbar__title, .v-app-bar .v-toolbar-title__placeholder {
                white-space: normal; font-size: 18px; line-height: 1.2;
            }
        """)
        with solara.ColumnsResponsive(default=[12], medium=[4, 8], gutters=False):
            with solara.Card('Setup'):
                SetupPanel(running, on_start=lambda: set_selected_run(None))
            with solara.Column(style={'min-width': '0', 'max-width': '100%'}):
                HistoryPanel(version, selected_run, set_selected_run, running, snap.run_id)
                if saved_error:
                    solara.Error(saved_error)
                if displayed is not None:
                    MonitorPanel(displayed)
                    if displayed.result is not None:
                        ResultsPanel(displayed.run_id, displayed.result)
                    elif selected_run:
                        solara.Info('This run has no saved result graphs. Legacy runs contain only '
                                    'loss history; failed or interrupted runs may have no test results.')
