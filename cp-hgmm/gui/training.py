"""Background per-class training service and persistent run history for the GUI."""

import json
import logging
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, TensorDataset

from models.lit_system_em import LitEMTorch

RUNS_DIR = Path(os.environ.get('CPHGMM_RUNS_DIR', 'runs')).absolute()
METRICS = ('aug_loss', 'nll', 'mean_frob_norm_sum', 'val_aug_loss', 'val_nll')
_RUN_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')
_RUNS_LOCK = threading.RLock()
_LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    dataset: str = 'Coffee'
    num_states: int = 9
    kappa: float = 5e3
    max_epochs: int = 1000
    batch_size: int = 1000
    patience: int = 3
    seed: int = 53
    dataset_source: str = 'ucr'
    dataset_name: str = ''  # Human-readable name; uploaded datasets use an opaque id in `dataset`.


@dataclass
class TrainingResult:
    """Test-set evaluation from one independently trained model per class."""
    checkpoint: str  # Compatibility alias for the first non-empty class checkpoint.
    y_test: np.ndarray
    y_pred: np.ndarray
    labels: np.ndarray
    sample_lpd: np.ndarray
    a_diff_norms: Dict[str, np.ndarray]
    test_elpd: float
    checkpoints: Dict[str, str] = field(default_factory=dict)


@dataclass
class JobSnapshot:
    run_id: Optional[str]
    status: str
    config: Optional[TrainingConfig]
    epoch: int  # Completed epochs for the current/most recently trained class.
    history: List[dict]
    log: List[str]
    error: Optional[str]
    started: Optional[float]
    finished: Optional[float]
    result: Optional[TrainingResult]
    version: int
    current_class: Optional[str] = None
    history_records: int = 0


class _MonitorCallback(Callback):
    def __init__(self, job: 'TrainingJob', class_name: str):
        self.job = job
        self.class_name = class_name

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        row = {'epoch': trainer.current_epoch, 'class': self.class_name}
        for name in METRICS:
            if name in metrics:
                row[name] = float(metrics[name])
        self.job._add_epoch(row)
        if self.job._stop_requested.is_set():
            trainer.should_stop = True


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_run_dir(run_id: str) -> Path:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f'Invalid run ID: {run_id!r}')
    if RUNS_DIR.is_symlink():
        raise ValueError(f'Runs path must not be a symbolic link: {RUNS_DIR}')
    root = RUNS_DIR.resolve()
    path = RUNS_DIR / run_id
    if path.is_symlink() or path.parent.resolve() != root:
        raise ValueError(f'Unsafe run directory for {run_id!r}')
    return path


def _timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def _save_result(run_dir: Path, result: TrainingResult):
    arrays = {
        'y_test': np.asarray(result.y_test),
        'y_pred': np.asarray(result.y_pred),
        'labels': np.asarray(result.labels),
        'sample_lpd': np.asarray(result.sample_lpd),
    }
    for name, value in arrays.items():
        if value.dtype.hasobject:
            arrays[name] = value.astype(str)
    continuity = {}
    for index, (class_name, values) in enumerate(result.a_diff_norms.items()):
        array_key = f'a_diff_{index}'
        arrays[array_key] = np.asarray(values)
        continuity[class_name] = array_key
    metadata = {
        'checkpoint': result.checkpoint,
        'checkpoints': result.checkpoints,
        'test_elpd': result.test_elpd,
        'continuity': continuity,
    }
    arrays['metadata_json'] = np.asarray(json.dumps(metadata))
    temporary = run_dir / f'.results-{uuid.uuid4().hex}.tmp'
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, run_dir / 'results.npz')
    finally:
        temporary.unlink(missing_ok=True)


def _load_result(run_dir: Path) -> Optional[TrainingResult]:
    path = run_dir / 'results.npz'
    if not path.exists():
        raise FileNotFoundError(f'Missing result arrays for run {run_dir.name!r}')
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Unsafe results file in {run_dir.name}')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json'].item()))
        a_diff_norms = {
            class_name: np.array(archive[array_key])
            for class_name, array_key in metadata['continuity'].items()
        }
        return TrainingResult(
            checkpoint=metadata.get('checkpoint', ''),
            checkpoints=dict(metadata.get('checkpoints', {})),
            y_test=np.array(archive['y_test']),
            y_pred=np.array(archive['y_pred']),
            labels=np.array(archive['labels']),
            sample_lpd=np.array(archive['sample_lpd']),
            a_diff_norms=a_diff_norms,
            test_elpd=float(metadata['test_elpd']),
        )


class TrainingJob:
    """Runs one training at a time in a background thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._version = 0
        self._reset(None, None, 'idle')

    def _reset(self, run_id, config, status):
        self._run_id = run_id
        self._config = config
        self._status = status
        self._history: List[dict] = []
        self._log: List[str] = []
        self._error = None
        self._started = None
        self._finished = None
        self._result = None
        self._current_class = None
        self._version += 1

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            progress_class = self._current_class
            if progress_class is None and self._history:
                progress_class = self._history[-1].get('class')
            class_epochs = [
                row['epoch'] + 1 for row in self._history
                if progress_class is not None and row.get('class') == progress_class
            ]
            return JobSnapshot(
                run_id=self._run_id, status=self._status, config=self._config,
                epoch=max(class_epochs, default=0), history=list(self._history),
                log=list(self._log), error=self._error, started=self._started,
                finished=self._finished, result=self._result, version=self._version,
                current_class=self._current_class,
                history_records=len(self._history),
            )

    def start(self, config: TrainingConfig) -> str:
        if config.dataset_source not in ('ucr', 'uploaded'):
            raise ValueError("dataset_source must be 'ucr' or 'uploaded'")
        with _RUNS_LOCK, self._lock:
            if self.is_running:
                raise RuntimeError('A training run is already in progress.')
            run_id = f"run-{datetime.now().strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:12]}"
            self._reset(run_id, config, 'loading')
            self._started = time.time()
            self._stop_requested.clear()
            self._thread = threading.Thread(target=self._run, name=f'train-{run_id}', daemon=True)
            self._thread.start()
            return run_id

    def stop(self):
        if self.is_running and not self._stop_requested.is_set():
            self._stop_requested.set()
            self._log_line('Stop requested; finishing the current epoch...')

    def _set(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, f'_{key}', value)
            self._version += 1

    def _log_line(self, msg: str):
        with self._lock:
            self._log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            self._version += 1

    def _add_epoch(self, row: dict):
        with self._lock:
            self._history.append(row)
            self._version += 1
        parts = ', '.join(f'{k}={v:.4g}' for k, v in row.items()
                          if k not in ('epoch', 'class'))
        self._log_line(f"Class {row['class']} epoch {row['epoch']}: {parts}")

    def _run(self):
        config = self._config
        run_dir = _safe_run_dir(self._run_id)
        try:
            from gui.datasets import load_dataset

            torch.manual_seed(config.seed)
            label = (f"'{config.dataset_name}' ({config.dataset})"
                     if config.dataset_name and config.dataset_name != config.dataset
                     else f"'{config.dataset}'")
            self._log_line(f'Loading {config.dataset_source} dataset {label}...')
            data = load_dataset(config.dataset_source, config.dataset)
            self._log_line(
                f'Loaded {len(data.dataset_train)} train / {len(data.dataset_test)} test samples, '
                f'length {data.data_len}, {data.output_dim} output(s), '
                f'classes {[str(c) for c in data.unique_labels]}.'
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            self._set(status='training')
            self._log_line('Training one model per class.')
            result = self._fit_and_evaluate(data, run_dir)
            if result is None:
                self._set(status='stopped', result=None, current_class=None, finished=time.time())
                self._log_line('Stopped before all class models were trained; no complete result is available.')
            else:
                self._set(status='finished', result=result, current_class=None, finished=time.time())
                self._log_line(f'Done. Test ELPD (average over all test classes) = {result.test_elpd:.4f}')
        except Exception as err:
            self._set(status='failed', error=f'{type(err).__name__}: {err}',
                      current_class=None, finished=time.time())
            self._log_line(traceback.format_exc())
        finally:
            try:
                self._save_summary(run_dir)
            except Exception as err:
                self._set(status='failed', error=f'Failed to save run: {type(err).__name__}: {err}',
                          finished=time.time())
                self._log_line(traceback.format_exc())

    def _fit_and_evaluate(self, data, run_dir: Path) -> Optional[TrainingResult]:
        train_x, train_y, train_labels = data.dataset_train.tensors
        test_x, test_y, test_labels = data.dataset_test.tensors
        test_count = len(test_labels)
        predictions = np.empty(tuple(test_y.squeeze(-1).shape), dtype=np.float64)
        sample_lpd = np.empty(test_count, dtype=np.float64)
        labels = data.label_encoder.inverse_transform(test_labels.numpy())
        a_diff_norms = {}
        checkpoints = {}
        y_test = test_y.squeeze(-1).numpy()

        for raw_label in data.unique_labels:
            class_name = str(raw_label)
            encoded_label = int(data.label_encoder.transform([raw_label])[0])
            self._set(current_class=class_name)
            if self._stop_requested.is_set():
                return None
            train_indices = torch.nonzero(train_labels == encoded_label, as_tuple=False).flatten()
            test_indices = torch.nonzero(test_labels == encoded_label, as_tuple=False).flatten()
            if train_indices.numel() == 0:
                raise ValueError(f'Class {class_name!r} has no training samples.')
            train_set = TensorDataset(train_x[train_indices], train_y[train_indices],
                                      train_labels[train_indices])
            if test_indices.numel() == 0:
                self._log_line(
                    f'Class {class_name} is absent from TEST; using its training samples for validation.'
                )
                val_set = train_set
            else:
                val_set = TensorDataset(test_x[test_indices], test_y[test_indices],
                                        test_labels[test_indices])
            train_loader = DataLoader(train_set, batch_size=self._config.batch_size)
            val_loader = DataLoader(val_set, batch_size=self._config.batch_size)
            model = LitEMTorch(
                length=data.data_len, num_states=self._config.num_states,
                num_outputs=data.output_dim, kappa=self._config.kappa,
            )
            class_dir = run_dir / 'checkpoints' / f'class-{len(checkpoints):03d}'
            best_path = model.fit(
                train_loader, val_loader, max_epochs=self._config.max_epochs,
                verbose=False, callbacks=[_MonitorCallback(self, class_name)],
                default_root_dir=str(class_dir), patience=self._config.patience,
            )
            if self._stop_requested.is_set():
                return None
            if best_path:
                model = LitEMTorch.load_from_checkpoint(best_path)
            checkpoints[class_name] = str(best_path or '')
            with torch.no_grad():
                test_y_tensor = test_y[test_indices]
                class_predictions = model.smoother(test_y_tensor)[2]
                class_lpd = model.smoother.log_pred_density(test_y_tensor).mean(dim=1)
                a_t = model.smoother.a_tensor
                if a_t.dim() > 2 and a_t.size(0) > 1:
                    continuity = torch.linalg.matrix_norm(a_t[:-1] - a_t[1:], ord='fro')
                else:
                    continuity = torch.zeros(0)
            predictions[test_indices.numpy()] = class_predictions.squeeze(-1).cpu().numpy()
            sample_lpd[test_indices.numpy()] = class_lpd.cpu().numpy()
            a_diff_norms[class_name] = continuity.cpu().numpy()
            self._log_line(f'Finished class {class_name} ({len(test_indices)} test samples).')
            if self._stop_requested.is_set():
                return None

        if self._stop_requested.is_set():
            return None
        return TrainingResult(
            checkpoint=next((path for path in checkpoints.values() if path), ''),
            checkpoints=checkpoints,
            y_test=y_test,
            y_pred=predictions,
            labels=np.asarray(labels).astype(str),
            sample_lpd=sample_lpd,
            a_diff_norms=a_diff_norms,
            test_elpd=float(sample_lpd.mean()),
        )

    def _save_summary(self, run_dir: Path):
        with _RUNS_LOCK:
            snap = self.snapshot()
            run_dir.mkdir(parents=True, exist_ok=True)
            if snap.result is not None:
                _save_result(run_dir, snap.result)
            val_losses = [r['val_aug_loss'] for r in snap.history if 'val_aug_loss' in r]
            summary = {
                'run_id': snap.run_id,
                'started': datetime.fromtimestamp(snap.started).isoformat(timespec='seconds')
                if snap.started else None,
                'finished': snap.finished,
                'status': snap.status,
                'config': asdict(snap.config) if snap.config else None,
                'epochs': snap.history_records,
                'current_epoch': snap.epoch,
                'best_val_aug_loss': min(val_losses) if val_losses else None,
                'test_elpd': snap.result.test_elpd if snap.result else None,
                'checkpoint': snap.result.checkpoint if snap.result else None,
                'checkpoints': snap.result.checkpoints if snap.result else {},
                'error': snap.error,
                'history': snap.history,
                'current_class': snap.current_class,
                'results_file': 'results.npz' if snap.result is not None else None,
            }
            _atomic_json(run_dir / 'summary.json', summary)
            self._set()


def list_runs() -> List[dict]:
    """List saved summaries newest first; malformed entries are logged and skipped."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    if RUNS_DIR.is_symlink() or not RUNS_DIR.is_dir():
        raise ValueError(f'Runs path is not a safe directory: {RUNS_DIR}')
    for path in RUNS_DIR.iterdir():
        if not path.is_dir() or path.is_symlink() or not _RUN_ID_RE.fullmatch(path.name):
            continue
        summary_path = path / 'summary.json'
        if not summary_path.exists():
            continue
        try:
            if summary_path.is_symlink() or not summary_path.is_file():
                raise ValueError('summary is not a regular file')
            with summary_path.open(encoding='utf-8') as stream:
                summary = json.load(stream)
            if not isinstance(summary, dict) or summary.get('run_id') != path.name:
                raise ValueError('summary run_id does not match its directory')
            runs.append(summary)
        except (OSError, ValueError, json.JSONDecodeError) as err:
            _LOGGER.warning('Skipping corrupt run summary %s: %s', summary_path, err)
    return sorted(runs, key=lambda run: run.get('started') or '', reverse=True)


def load_run(run_id: str) -> JobSnapshot:
    """Load a persisted run. Legacy summary-only runs have ``result=None``."""
    run_dir = _safe_run_dir(run_id)
    summary_path = run_dir / 'summary.json'
    if not summary_path.exists() or summary_path.is_symlink() or not summary_path.is_file():
        raise FileNotFoundError(f'No saved summary for run {run_id!r}')
    try:
        with summary_path.open(encoding='utf-8') as stream:
            summary = json.load(stream)
        if not isinstance(summary, dict) or summary.get('run_id') != run_id:
            raise ValueError('summary run_id does not match its directory')
        config_data = summary.get('config')
        config = TrainingConfig(**config_data) if config_data else None
        result = _load_result(run_dir) if summary.get('results_file') else None
        started = _timestamp(summary.get('started'))
        finished = _timestamp(summary.get('finished'))
        return JobSnapshot(
            run_id=run_id,
            status=summary.get('status', 'unknown'),
            config=config,
            epoch=int(summary.get('current_epoch', 0)),
            history=list(summary.get('history') or []),
            log=[],
            error=summary.get('error'),
            started=started,
            finished=finished,
            result=result,
            version=0,
            current_class=summary.get('current_class'),
            history_records=int(summary.get('epochs', len(summary.get('history', [])))),
        )
    except Exception as err:
        raise ValueError(f'Could not load run {run_id!r}: {err}') from err


def delete_run(run_id: str):
    """Delete one saved run, refusing to remove the active training run."""
    run_dir = _safe_run_dir(run_id)
    with _RUNS_LOCK, job._lock:
        if job.is_running and job._run_id == run_id:
            raise RuntimeError('Cannot delete the active training run.')
        if run_dir.exists():
            if run_dir.is_symlink() or not run_dir.is_dir():
                raise ValueError(f'Unsafe run directory for {run_id!r}')
            shutil.rmtree(run_dir)
        if job._run_id == run_id:
            job._reset(None, None, 'idle')
            job._thread = None
        else:
            job._version += 1


def delete_all_runs() -> List[str]:
    """Delete all saved runs; refused while any training thread is active."""
    with _RUNS_LOCK, job._lock:
        if job.is_running:
            raise RuntimeError('Cannot delete runs while training is active.')
        deleted = []
        if not RUNS_DIR.exists():
            return deleted
        if RUNS_DIR.is_symlink() or not RUNS_DIR.is_dir():
            raise ValueError(f'Runs path is not a safe directory: {RUNS_DIR}')
        for path in RUNS_DIR.iterdir():
            if not _RUN_ID_RE.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
                continue
            shutil.rmtree(path)
            deleted.append(path.name)
        if job._run_id is not None:
            job._reset(None, None, 'idle')
            job._thread = None
        else:
            job._version += 1
        return deleted


job = TrainingJob()
