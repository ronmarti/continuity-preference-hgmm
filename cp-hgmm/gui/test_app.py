"""Plot and component regression tests, without a browser or archive downloads."""
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import ipyvuetify as v
import solara

from gui import app


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.result = SimpleNamespace(
            checkpoint='class-models',
            checkpoints={'a': 'a.ckpt', 'b': 'b.ckpt'},
            y_test=np.zeros((4, 5, 1)),
            y_pred=np.ones((4, 5, 1)),
            labels=np.array(['a', 'b', 'a', 'b']),
            sample_lpd=np.array([-5., -4., -3., -2.]),
            a_diff_norms={'a': np.array([1., 2.]), 'b': np.array([3., 4.])},
            test_elpd=-3.5,
        )

    def test_density_plot_states_direction(self):
        fig = app._model_figure(self.result)
        self.assertEqual(len(fig.axes), 1)
        self.assertIn('higher is better', fig.axes[0].get_ylabel())
        self.assertEqual([tick.get_text() for tick in fig.axes[0].get_xticklabels()], ['a', 'b'])

    def test_continuity_figures_use_distinct_class_arrays(self):
        for label, norms in self.result.a_diff_norms.items():
            fig = app._continuity_figure(label, norms)
            self.assertIn(f'Class {label}', fig.axes[0].get_title())
            np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), norms)

    def test_time_invariant_continuity_is_explicit(self):
        fig = app._continuity_figure('a', np.array([]))
        self.assertEqual(len(fig.axes[0].lines), 0)
        self.assertIn('time-invariant', fig.axes[0].texts[0].get_text())

    def test_losses_do_not_join_class_boundaries(self):
        fig = app._loss_figure([
            {'class': 'a', 'epoch': 0, 'aug_loss': 3., 'mean_frob_norm_sum': 1.},
            {'class': 'a', 'epoch': 1, 'aug_loss': 2., 'mean_frob_norm_sum': .5},
            {'class': 'b', 'epoch': 0, 'aug_loss': 8., 'mean_frob_norm_sum': 4.},
        ])
        self.assertEqual(len(fig.axes[0].lines), 2)
        self.assertEqual(len(fig.axes[1].lines), 2)
        self.assertEqual([line.get_label() for line in fig.axes[0].lines],
                         ['Class a: aug_loss', 'Class b: aug_loss'])

    def test_legacy_loss_history_still_plots(self):
        fig = app._loss_figure([{'epoch': 0, 'aug_loss': 2.}])
        self.assertEqual(fig.axes[0].lines[0].get_label(), 'aug_loss')

    def test_selecting_saved_runs_refreshes_loss_plot_with_same_version(self):
        snapshot = SimpleNamespace(
            run_id='first', status='finished', started=1., finished=2., epoch=1,
            config=SimpleNamespace(max_epochs=1), current_class=None, error=None,
            log=[], version=0, history=[{'epoch': 0, 'aug_loss': 2.}],
        )
        with patch.object(app, '_loss_figure', wraps=app._loss_figure) as plot:
            _, context = solara.render(app.MonitorPanel(snapshot), handle_error=False)
            self.addCleanup(context.close)
            other = SimpleNamespace(**dict(vars(snapshot), run_id='second',
                                           history=[{'epoch': 0, 'aug_loss': 7.}]))
            context.render(app.MonitorPanel(other))
            self.assertEqual(plot.call_count, 2)
            self.assertEqual(plot.call_args.args[0], other.history)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.run = {
            'run_id': 'saved-run', 'status': 'finished',
            'config': {'dataset': 'Coffee', 'num_states': 1, 'kappa': 0},
            'epochs': 1, 'best_val_aug_loss': 1., 'test_elpd': -1.,
        }

    def render_history(self, running=False, selected='saved-run'):
        selected_callback = unittest.mock.Mock()
        widget, context = solara.render(
            app.HistoryPanel(1, selected, selected_callback, running, 'active-run'),
            handle_error=False,
        )
        self.addCleanup(context.close)
        return widget, context, selected_callback

    def button(self, context, label):
        buttons = context.find(v.Btn).widgets
        return next(button for button in buttons if label in button.children)

    @patch.object(app, 'list_runs')
    @patch.object(app, 'delete_run')
    def test_delete_requires_confirmation(self, delete, list_runs):
        list_runs.return_value = [self.run]
        _, context, selected_callback = self.render_history()
        button = self.button(context, 'Delete selected run')
        self.assertTrue(button.disabled)
        context.find(v.Checkbox).widget.v_model = True
        button = self.button(context, 'Delete selected run')
        self.assertFalse(button.disabled)
        button.fire_event('click', {})
        delete.assert_called_once_with('saved-run')
        selected_callback.assert_called_once_with(None)

    @patch.object(app, 'list_runs')
    def test_all_deletion_disabled_during_training(self, list_runs):
        list_runs.return_value = [self.run]
        _, context, _ = self.render_history(running=True)
        context.find(v.Checkbox).widget.v_model = True
        self.assertTrue(self.button(context, 'Delete all runs').disabled)
        self.assertFalse(self.button(context, 'Delete selected run').disabled)

    @patch.object(app, 'list_runs')
    def test_selected_active_run_cannot_be_deleted(self, list_runs):
        list_runs.return_value = [dict(self.run, run_id='active-run')]
        _, context, _ = self.render_history(running=True, selected='active-run')
        context.find(v.Checkbox).widget.v_model = True
        self.assertTrue(self.button(context, 'Delete selected run').disabled)

    @patch.object(app, 'list_uploaded_datasets',
                  return_value=[{'id': 'abc123', 'name': 'Old upload'}])
    @patch.object(app, 'list_runs')
    def test_dataset_column_shows_display_name(self, list_runs, _):
        def uploaded(run_id, **config):
            return dict(self.run, run_id=run_id,
                        config=dict(self.run['config'], dataset_source='uploaded', **config))

        list_runs.return_value = [
            uploaded('named', dataset='f00d', dataset_name='My data'),
            uploaded('legacy', dataset='abc123'),
            uploaded('gone', dataset='zzz'),
            self.run,
        ]
        _, context, _ = self.render_history(selected=None)
        names = [row['dataset'] for row in context.find(app.RunTableWidget).widget.items]
        self.assertEqual(names, ['My data', 'Old upload', 'zzz (deleted upload)', 'Coffee'])

    @patch.object(app, 'list_runs')
    def test_run_selection_calls_parent(self, list_runs):
        list_runs.return_value = [self.run]
        _, context, callback = self.render_history(selected=None)
        table = context.find(app.RunTableWidget).widget
        self.assertIsNone(table.selected)
        self.assertEqual(table.items[0]['run_id'], 'saved-run')
        table.vue_select_run('saved-run')
        callback.assert_called_once_with('saved-run')

    @patch.object(app, 'list_runs')
    def test_row_deselection_returns_to_live(self, list_runs):
        list_runs.return_value = [self.run, dict(self.run, run_id='other-run', test_elpd=float('nan'))]
        _, context, callback = self.render_history(selected='saved-run')
        table = context.find(app.RunTableWidget).widget
        self.assertEqual(table.selected, 'saved-run')
        self.assertIsNone(table.items[1]['test_elpd'])
        table.vue_select_run(None)
        callback.assert_called_once_with(None)

    @patch.object(app, 'list_runs')
    def test_show_live_run_clears_selection(self, list_runs):
        list_runs.return_value = [self.run]
        _, context, callback = self.render_history()
        self.button(context, 'Show live run').fire_event('click', {})
        callback.assert_called_once_with(None)

    @patch.object(app, 'load_run', side_effect=ValueError('missing results'))
    def test_load_error_is_visible_not_live_fallback(self, load):
        snapshot, error = app._read_saved_run('saved-run')
        self.assertIsNone(snapshot)
        self.assertIn('missing results', error)


class DatasetControlsTests(unittest.TestCase):
    @patch.object(app, 'list_uploaded_datasets',
                  return_value=[{'id': 'dataset-1', 'name': 'My dataset'}])
    @patch.object(app.job, 'start')
    def test_select_uploaded_dataset_and_start(self, start, list_datasets):
        on_start = unittest.mock.Mock()
        _, context = solara.render(app.SetupPanel(False, on_start), handle_error=False)
        self.addCleanup(context.close)
        values = {label: context.find(v.TextField, label=label).widget.v_model for label in (
            'Max epochs', 'Batch size', 'Early-stopping patience', 'Random seed',
        )}
        context.find(v.Select, label='Dataset source').widget.v_model = 'Uploaded dataset'
        for label, value in values.items():
            self.assertEqual(context.find(v.TextField, label=label).widget.v_model, value, label)
        context.find(v.Select, label='Uploaded dataset').widget.v_model = 'My dataset'
        button = next(b for b in context.find(v.Btn).widgets if 'Start training' in b.children)
        button.fire_event('click', {})
        config = start.call_args.args[0]
        self.assertEqual(config.dataset_source, 'uploaded')
        self.assertEqual(config.dataset, 'dataset-1')
        self.assertEqual(config.dataset_name, 'My dataset')
        on_start.assert_called_once_with()

    def test_hidden_states_value_is_always_visible(self):
        self.addCleanup(setattr, app.num_states, 'value', app.num_states.value)
        _, context = solara.render(app.SetupPanel(False), handle_error=False)
        self.addCleanup(context.close)
        app.num_states.value = 7
        slider = context.find(v.Slider).widget
        self.assertEqual(slider.label, 'Hidden states: 7')
        self.assertFalse(slider.thumb_label)

    @patch.object(app, 'save_uploaded_dataset', return_value='dataset-2')
    def test_upload_passes_both_files_and_selects_saved_dataset(self, save):
        callbacks = {}

        @solara.component
        def FakeDrop(label, on_file):
            callbacks[label] = on_file
            solara.Text(label)

        saved = unittest.mock.Mock()
        with patch.object(app.solara, 'FileDrop', FakeDrop):
            _, context = solara.render(app.DatasetUpload(saved), handle_error=False)
            self.addCleanup(context.close)
            context.find(v.TextField, label='Dataset display name').widget.v_model = 'Uploaded'
            for split in ('TRAIN', 'TEST'):
                callbacks[f'Drop {split} .ts file here']({
                    'name': f'{split}.ts', 'size': 4, 'file_obj': io.BytesIO(b'data'),
                })
            button = next(b for b in context.find(v.Btn).widgets
                          if 'Validate and save dataset' in b.children)
            self.assertFalse(button.disabled)
            button.fire_event('click', {})
            save.assert_called_once_with('Uploaded', 'TRAIN.ts', b'data', 'TEST.ts', b'data')
            saved.assert_called_once_with('dataset-2')


if __name__ == '__main__':
    unittest.main()
