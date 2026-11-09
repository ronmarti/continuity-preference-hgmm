import json
import shutil
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from gui import training


class _Encoder:
    classes_ = np.array(['alpha', 'beta'])

    def transform(self, labels):
        return np.array([int(label == 'beta') for label in labels])

    def inverse_transform(self, labels):
        return self.classes_[np.asarray(labels, dtype=int)]


class _Data:
    def __init__(self):
        self.label_encoder = _Encoder()
        self.unique_labels = self.label_encoder.classes_
        self.data_len = 3
        self.output_dim = 1
        train_labels = torch.tensor([0, 1, 0, 1])
        test_labels = torch.tensor([1, 0, 1, 0])
        train_y = torch.arange(12, dtype=torch.double).reshape(4, 3, 1, 1)
        test_y = torch.arange(12, 24, dtype=torch.double).reshape(4, 3, 1, 1)
        train_x = torch.arange(3, dtype=torch.double).repeat(4, 1)
        test_x = torch.arange(3, dtype=torch.double).repeat(4, 1)
        self.dataset_train = TensorDataset(train_x, train_y, train_labels)
        self.dataset_test = TensorDataset(test_x, test_y, test_labels)


class _Smoother:
    def __init__(self, tag):
        self.tag = tag
        self.a_tensor = torch.arange(12, dtype=torch.double).reshape(3, 2, 2)

    def __call__(self, values):
        return None, None, values + self.tag

    def log_pred_density(self, values):
        return torch.full(values.shape[:2], self.tag, dtype=values.dtype)


class _FakeModel:
    instances = []

    def __init__(self, length, num_states, num_outputs, kappa):
        self.smoother = None
        self.__class__.instances.append(self)

    def fit(self, train_loader, val_loader, **kwargs):
        train_labels = torch.cat([batch[2] for batch in train_loader])
        val_labels = torch.cat([batch[2] for batch in val_loader])
        self.class_id = int(train_labels[0])
        assert torch.all(train_labels == self.class_id)
        assert torch.all(val_labels == self.class_id)
        self.smoother = _Smoother(10 * (self.class_id + 1))
        callback = kwargs['callbacks'][0]
        callback.on_train_epoch_end(
            type('Trainer', (), {
                'callback_metrics': {'val_aug_loss': torch.tensor(0.5)},
                'current_epoch': 0,
            })(),
            self,
        )
        return ''


class TrainingBackendTests(unittest.TestCase):
    def setUp(self):
        self.run_root = Path(__file__).parent / f'.training-test-runs-{uuid.uuid4().hex}'
        self.run_root.mkdir()
        self._runs_dir = patch.object(training, 'RUNS_DIR', self.run_root)
        self._runs_dir.start()

    def tearDown(self):
        self._runs_dir.stop()
        shutil.rmtree(self.run_root, ignore_errors=True)

    def test_one_model_per_class_and_original_test_order(self):
        data = _Data()
        _FakeModel.instances = []
        job = training.TrainingJob()
        job._config = training.TrainingConfig(batch_size=2)
        run_dir = self.run_root / 'model-run'
        with patch.object(training, 'LitEMTorch', _FakeModel):
            result = job._fit_and_evaluate(data, run_dir)

        self.assertEqual(len(_FakeModel.instances), 2)
        self.assertEqual(set(result.a_diff_norms), {'alpha', 'beta'})
        np.testing.assert_array_equal(result.labels, ['beta', 'alpha', 'beta', 'alpha'])
        expected = data.dataset_test.tensors[1].squeeze(-1).numpy().copy()
        expected[[1, 3]] += 10
        expected[[0, 2]] += 20
        np.testing.assert_array_equal(result.y_pred, expected)
        np.testing.assert_array_equal(result.sample_lpd, [20, 10, 20, 10])
        self.assertEqual([row['class'] for row in job.snapshot().history], ['alpha', 'beta'])
        self.assertEqual(job.snapshot().epoch, 1)
        self.assertEqual(job.snapshot().history_records, 2)

    def test_class_missing_from_test_still_trains(self):
        data = _Data()
        x, y, _ = data.dataset_test.tensors
        data.dataset_test = TensorDataset(x[:2], y[:2], torch.tensor([0, 0]))
        job = training.TrainingJob()
        job._config = training.TrainingConfig(batch_size=2)
        _FakeModel.instances = []
        with patch.object(training, 'LitEMTorch', _FakeModel):
            result = job._fit_and_evaluate(data, self.run_root / 'missing-test-class')
        self.assertEqual(len(_FakeModel.instances), 2)
        self.assertEqual(result.labels.tolist(), ['alpha', 'alpha'])
        self.assertEqual(set(result.a_diff_norms), {'alpha', 'beta'})
        self.assertTrue(any('absent from TEST' in line for line in job.snapshot().log))

    def test_stop_does_not_return_partial_results_as_complete(self):
        data = _Data()
        job = training.TrainingJob()
        job._config = training.TrainingConfig(batch_size=2)
        original_fit = _FakeModel.fit

        def stop_after_epoch(model, *args, **kwargs):
            output = original_fit(model, *args, **kwargs)
            job._stop_requested.set()
            return output

        with patch.object(training, 'LitEMTorch', _FakeModel), \
                patch.object(_FakeModel, 'fit', stop_after_epoch):
            result = job._fit_and_evaluate(data, self.run_root / 'stopped')
        self.assertIsNone(result)
        self.assertEqual([row['class'] for row in job.snapshot().history], ['alpha'])

    def test_result_round_trip_and_legacy_summary(self):
        run_id = 'saved-run'
        run_dir = self.run_root / run_id
        run_dir.mkdir()
        result = training.TrainingResult(
            checkpoint='best.ckpt',
            y_test=np.ones((2, 3, 1)),
            y_pred=np.zeros((2, 3, 1)),
            labels=np.array(['alpha', 'beta']),
            sample_lpd=np.array([-1., -2.]),
            a_diff_norms={'alpha': np.array([0.1]), 'beta': np.array([0.2])},
            test_elpd=-1.5,
            checkpoints={'alpha': 'a.ckpt', 'beta': 'b.ckpt'},
        )
        training._save_result(run_dir, result)
        (run_dir / 'summary.json').write_text(json.dumps({
            'run_id': run_id, 'started': '2025-01-01T00:00:00',
            'status': 'finished', 'config': {'dataset': 'Coffee'},
            'epochs': 1, 'history': [{'epoch': 0, 'class': 'alpha'}],
            'results_file': 'results.npz',
        }), encoding='utf-8')
        loaded = training.load_run(run_id)
        self.assertEqual(loaded.config.dataset_source, 'ucr')
        self.assertEqual(loaded.history[0]['class'], 'alpha')
        np.testing.assert_array_equal(loaded.result.labels, result.labels)
        np.testing.assert_array_equal(loaded.result.a_diff_norms['beta'], [0.2])
        self.assertEqual(loaded.result.checkpoints['alpha'], 'a.ckpt')

        legacy_id = 'legacy-run'
        legacy_dir = self.run_root / legacy_id
        legacy_dir.mkdir()
        (legacy_dir / 'summary.json').write_text(json.dumps({
            'run_id': legacy_id, 'started': '2024-01-02T03:04:05',
            'status': 'finished', 'config': {'dataset': 'Old'},
            'epochs': 0, 'history': [],
        }), encoding='utf-8')
        legacy = training.load_run(legacy_id)
        self.assertIsNone(legacy.result)
        self.assertIsNotNone(legacy.started)
        self.assertIsNone(legacy.finished)
        self.assertIsNone(legacy.current_class)
        self.assertEqual(legacy.epoch, 0)

    def test_corrupt_summary_is_logged_and_load_error_is_explicit(self):
        run_dir = self.run_root / 'broken-run'
        run_dir.mkdir()
        (run_dir / 'summary.json').write_text('{', encoding='utf-8')
        with self.assertLogs(training._LOGGER, level='WARNING'):
            self.assertEqual(training.list_runs(), [])
        with self.assertRaisesRegex(ValueError, 'Could not load run'):
            training.load_run(run_dir.name)

    def test_delete_is_confined_and_active_run_is_protected(self):
        run_dir = self.run_root / 'removable'
        run_dir.mkdir()
        old_root = training.job
        training.job = training.TrainingJob()
        try:
            training.job._run_id = 'active'
            active = threading.Event()
            training.job._thread = threading.Thread(target=active.wait)
            training.job._thread.start()
            with self.assertRaisesRegex(RuntimeError, 'active training'):
                training.delete_run('active')
            with self.assertRaisesRegex(RuntimeError, 'training is active'):
                training.delete_all_runs()
            active.set()
            training.job._thread.join()
            training.job._thread = None
            training.delete_run(run_dir.name)
            self.assertFalse(run_dir.exists())
            latest = self.run_root / 'latest'
            latest.mkdir()
            training.job._run_id = 'latest'
            training.job._result = object()
            training.delete_run('latest')
            self.assertIsNone(training.job.snapshot().run_id)
            self.assertIsNone(training.job.snapshot().result)
            self.assertEqual(training.job.snapshot().status, 'idle')
            with self.assertRaises(ValueError):
                training.delete_run('../outside')
        finally:
            if training.job._thread and training.job._thread.is_alive():
                active.set()
                training.job._thread.join()
            training.job = old_root

    def test_deleting_non_live_run_bumps_version_for_history_refresh(self):
        old_job = training.job
        training.job = training.TrainingJob()
        try:
            live_dir = self.run_root / 'live'
            old_dir = self.run_root / 'older'
            live_dir.mkdir()
            old_dir.mkdir()
            training.job._run_id = 'live'
            training.job._status = 'finished'
            version_before = training.job.version

            training.delete_run('older')

            self.assertFalse(old_dir.exists())
            self.assertTrue(live_dir.exists())
            self.assertEqual(training.job.snapshot().run_id, 'live')
            self.assertEqual(training.job.snapshot().status, 'finished')
            self.assertGreater(training.job.version, version_before)

            all_version_before = training.job.version
            training.delete_all_runs()
            self.assertFalse(live_dir.exists())
            self.assertIsNone(training.job.snapshot().run_id)
            self.assertGreater(training.job.version, all_version_before)
        finally:
            training.job = old_job


if __name__ == '__main__':
    unittest.main()
