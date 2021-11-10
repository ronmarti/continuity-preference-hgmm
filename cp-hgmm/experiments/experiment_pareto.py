from typing import Any, Iterable, Optional
from ax import core
from ax.core import optimization_config
from ax.core.data import Data
from ax.core.objective import MultiObjective
from pandas.core.frame import DataFrame
import torch

from ax.service.ax_client import AxClient
from ax.modelbridge.factory import get_MOO_EHVI

from ax import Experiment

from ax.service.utils.instantiation import make_search_space
from ax.service.utils.instantiation import make_objective_thresholds

from models.utils.dataset_torch_preparator import DatasetContainer
from experiments.regression_problem import RegressionProblem
import experiments.browse_dlg as browse_dlg

import pandas as pd
from ax import Models, Metric, Runner

from ax.service.utils.report_utils import exp_to_df

from tqdm import trange


def load_experiment_into_client(load_path, ax_client: AxClient):
    """
    Enriches the experiment with already known results from the given json containing
    ax client.
    """
    restored_ax_client = AxClient.load_from_json_file(load_path)
    restored_exp = restored_ax_client.experiment
    exp_data = restored_exp.fetch_data()
    for arm_name, arm in restored_exp.arms_by_name.items():
        params = arm.parameters
        obj_rows = exp_data.df.loc[exp_data.df['arm_name'] == arm_name]
        metrics = {
            row["metric_name"]: (row["mean"], row["sem"])
            for _, row in obj_rows.iterrows()
        }
        _, trial_index = ax_client.attach_trial(params)
        ax_client.complete_trial(trial_index=trial_index, raw_data=metrics)
    return ax_client


def load_preceding_experiment_data(exp: Experiment, load_path):
    """
    Enriches the experiment with already known results from the given csv
    containing experiment data.
    """
    loaded_data_df: DataFrame = pd.read_csv(load_path)
    loaded_data: Data = Data(loaded_data_df)
    exp.attach_data(loaded_data)
    # for row in loaded_data.iterrows():
    #     params = {
    #         'kappa': row['kappa'],
    #         'num_states': row['num_states']
    #     }
    #     exp.attach_data()
    #     obj_rows = exp_data.df.loc[exp_data.df['arm_name'] == arm_name]
    #     metrics = {
    #         row["metric_name"]: (row["mean"], row["sem"])
    #         for _, row in obj_rows.iterrows()
    #     }
    #     _, trial_index = ax_client.attach_trial(params)
    #     ax_client.complete_trial(trial_index=trial_index, raw_data=metrics)
    # return ax_client


def get_zero_parameters(labels: Iterable, num_states=2):
    kappa = 0.
    params = {
        problem.get_param_name('num_states', label): num_states
        for label in labels
    }
    params_kappa = {
        problem.get_param_name('kappa', label): kappa
        for label in labels
    }
    params = dict(params, **params_kappa)
    return params


def load_preceding_clients_experiments(ax_client, root='.'):
    preceding_experiments_collected = []
    while restore_filepath := browse_dlg.gui_fname(root):
        preceding_experiments_collected.append(restore_filepath)
    for filepath in preceding_experiments_collected:
        load_experiment_into_client(filepath, ax_client)


def load_preceding_experiments(experiment: Experiment, root='.'):
    preceding_experiments_collected = []
    while restore_filepath := browse_dlg.gui_fname(root):
        preceding_experiments_collected.append(restore_filepath.decode())
    for filepath in preceding_experiments_collected:
        load_preceding_experiment_data(experiment, filepath)


class ParetoOptimization():

    def __init__(
        self,
        problem,
        parameters,
        objectives,
        objectives_thresholds
    ) -> None:
        self.problem = problem
        parameters = parameters
        objectives = objectives
        objectives_thresholds = objectives_thresholds

    @property
    def utility_function(self):
        return self.problem.utility_function

    def initialize_experiment(experiment: Experiment, n_init):
        sobol = Models.SOBOL(search_space=experiment.search_space)

        for _ in range(n_init):
            trial = experiment.new_trial(sobol.gen(1))
            trial.arm.parameters
            trial.run()

        return experiment.fetch_data()


class MyMetric(Metric):
    def __init__(self, name: str, lower_is_better: Optional[bool] = False):
        super().__init__(name, lower_is_better)

    def fetch_trial_data(self, trial: core.base_trial.BaseTrial, **kwargs: Any) -> Data:
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            metric_data = trial.run_metadata[arm_name][0][self.name]
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": metric_data[0],  # mean value of this metric when this arm is used
                "sem": metric_data[1],  # standard error of the above mean
                "trial_index": trial.index,
            })
        data = Data(df=pd.DataFrame.from_records(records))
        return data


class FooRunner(Runner):
    def __init__(self, problem):
        self.problem = problem

    def run(self, trial):
        name_to_params = {
            arm.name: arm.parameters for arm in trial.arms
        }
        results = {arm_name: problem.utility_function(params) for arm_name, params in name_to_params.items()}

        return results

    @property
    def staging_required(self):
        return False


def export_experiment(experiment, full_path):
    df = exp_to_df(experiment)
    metrics_detailed = experiment.fetch_data().df

    unique_metrics = metrics_detailed['metric_name'].unique()

    aggregate = df.set_index('trial_index').copy()
    for metric_name in unique_metrics:
        metric_df = metrics_detailed[metrics_detailed['metric_name']==metric_name].set_index('trial_index')
        metric_df.drop(['metric_name'], inplace=True, axis=1)
        metric_df.rename(columns={'mean': f'{metric_name}_mean', 'sem': f'{metric_name}_sem'}, inplace=True)
        aggregate = pd.merge(aggregate, metric_df, left_index=True, right_index=True)
    # gap_general_df = metrics_detailed[metrics_detailed['metric_name']=='gap_pos_tr_val_elpd'].set_index('arm_name')

    aggregate.to_csv(full_path)


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    print(f'Run {__file__}')
    N_BATCH = 40

    # dataset_name = 'KR2700'
    # train_load_path = Path('data_for_tests/RobotMBConsumption/RobotMBConsumption_ALL.ts')
    # test_load_path = None

    # dset_container = DatasetContainer(dataset_name,
    #                                   mode='ts',
    #                                   test_ratio=0.98,
    #                                   dtype=torch.double,
    #                                   train=train_load_path,
    #                                   test=test_load_path)

    # CricketX, Epilepsy
    dataset_name = 'SonyAIBORobotSurface1'
    dset_container = DatasetContainer(dataset_name, mode='ucr', test_ratio=0.99)
    dset_container.plot_dataset()

    now_str = datetime.now().strftime('%Y%m%dT%H%M%S')
    experiment_root = Path(f'.out/experiments/pareto_{dataset_name}')
    experiment_root.mkdir(parents=True, exist_ok=True)
    filename = Path(f'{dataset_name}_{now_str}.csv')

    problem = RegressionProblem(
        dset_container,
        sample_length=dset_container.data_len,
        num_outputs=dset_container.output_dim,
        max_epochs=200,
        n_trials=10,
        k_folds=5,
        verbose=False
    )

    params = [
        {
            'name': 'num_states',
            'type': 'fixed',
            'value': 5,
            # 'bounds': [2, 15],
            'value_type': 'int',
        },
        {
            'name': 'kappa_sqrt',
            'type': 'range',
            'bounds': [0, 30],
            'value_type': 'float'
        }
    ]

    objectives = {
        'gap_pos_tr_val_elpd': 'minimize',
        'gap_elpd': 'maximize'
    }

    objective_thresholds = make_objective_thresholds([
        f'gap_pos_tr_val_elpd <= 0.9',
        f'gap_elpd >= -0.1',
    ], status_quo_defined=False)

    search_space = make_search_space(params, parameter_constraints=[])

    my_runner = FooRunner(problem=problem)
    objectives_by_metrics = [
        MyMetric(name='gap_pos_tr_val_elpd', lower_is_better=objectives['gap_pos_tr_val_elpd']=='minimize'),
        MyMetric(name='gap_elpd', lower_is_better=objectives['gap_elpd']=='minimize')
    ]

    mo = MultiObjective(metrics=objectives_by_metrics)
    opt_config = optimization_config.MultiObjectiveOptimizationConfig(
        objective=mo,
        objective_thresholds=objective_thresholds,
    )
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=opt_config,
        runner=my_runner,
    )

    # load_preceding_experiments(experiment, root=experiment_root)

    full_path = str(experiment_root / filename)
    full_path_experiment_raw = str(experiment_root / f'raw_{filename}')
    full_path_experiment_arms = str(experiment_root / f'arms_{filename}')

    sobol = Models.SOBOL(search_space=experiment.search_space)
    for _ in trange(5):
        trial = experiment.new_trial(sobol.gen(1))
        trial.run()
        trial.mark_completed()
        export_experiment(experiment, full_path)
        exp_to_df(experiment).to_csv(full_path_experiment_raw)
        experiment.fetch_data().df.to_csv(full_path_experiment_arms)
    ehvi_data = experiment.fetch_data()

    for _ in trange(N_BATCH):
        ehvi_model = get_MOO_EHVI(
            experiment=experiment,
            data=ehvi_data,
        )
        generator_run = ehvi_model.gen(1)
        trial = experiment.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()
        ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])
        export_experiment(experiment, full_path)
        exp_to_df(experiment).to_csv(full_path_experiment_raw)
        experiment.fetch_data().df.to_csv(full_path_experiment_arms)

    print(f'Results stored at {full_path}.')
    print('Use the visualization_pareto.ipynb notebook for viewing the stored experiment output.')
    print('Done')
else:
    print(f'Importing {__file__}')
