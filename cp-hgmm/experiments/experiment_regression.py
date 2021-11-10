from typing import Iterable
import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data import TensorDataset

from ax.service.managed_loop import optimize
from ax.storage.json_store.save import save_experiment
from ax.service.ax_client import AxClient

from models.utils.dataset_torch_preparator import DatasetContainer
from experiments.regression_problem import RegressionProblem
import experiments.browse_dlg as browse_dlg


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


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    print(f'Run {__file__}')

    # dataset_name = 'KR2700'
    # train_load_path = Path('data_for_tests/RobotMBConsumption/RobotMBConsumption_TRAIN.ts')
    # test_load_path = Path('data_for_tests/RobotMBConsumption/RobotMBConsumption_TEST.ts')
    # train_load_path = Path('N:/Datasets/TimeSeries/classification/IndustrialRobots/RobotMBConsumption/RobotMBConsumption_TRAIN.ts')
    # test_load_path = Path('N:/Datasets/TimeSeries/classification/IndustrialRobots/RobotMBConsumption/RobotMBConsumption_TEST.ts')
    dataset_name = 'KR5'
    train_load_path = Path(r'N:\Datasets\TimeSeries\classification\IndustrialRobots\RobotKR5Consumption\RobotKR5Consumption_TRAIN.ts')
    test_load_path = None

    dataset_container = DatasetContainer(dataset_name,
                                         mode='ts',
                                         dtype=torch.double,
                                         train=train_load_path,
                                         test=test_load_path,
                                         test_ratio=0.99)

    # dataset_name = 'SonyAIBORobotSurface2'
    # dataset_container = DatasetContainer(
    #     dataset_name,
    #     mode='ucr',
    #     test_ratio=0.99
    # )
    dataset_container.plot_dataset()

    now_str = datetime.now().strftime('%Y%m%dT%H%M%S')
    experiment_root = Path(f'.out/experiments/{dataset_name}')
    experiment_root.mkdir(parents=True, exist_ok=True)
    filename = Path(f'{dataset_name}_{now_str}.json')

    problem = RegressionProblem(
        dataset_container,
        sample_length=dataset_container.data_len,
        num_outputs=dataset_container.output_dim,
        max_epochs=100,
        n_trials=1,
        k_folds=10,
        verbose=False
    )

    ax_client = AxClient()
    ax_client.create_experiment(
        name=f'{dataset_name}_optimal_parameters_search',
        parameters=[
            {
                'name': 'num_states',
                'type': 'range',
                'bounds': [3, 9],
                'value_type': 'int'
            },
            {
                'name': 'kappa_sqrt',
                'type': 'range',
                'bounds': [0, 1e5],
                'value_type': 'float'
            }
        ],
        objective_name='gaps',
        minimize=False,
        parameter_constraints=[]
    )

    root = Path(f'.out/experiments/regr_{dataset_name}')
    load_preceding_clients_experiments(ax_client, root=root)

    full_path = str(experiment_root / filename)

    parameters_zero = [
        get_zero_parameters(problem.unique_labels, num_states=num_states)
        for num_states in range(9, 10)
    ]
    for param_zero in parameters_zero:
        parameters, trial_index = ax_client.attach_trial(param_zero)
        result, metadata = problem.utility_function(parameters)
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=result, metadata=metadata)
        ax_client.save_to_json_file(full_path)
    for i in range(40):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        result, metadata = problem.utility_function(parameters)
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=result, metadata=metadata)
        ax_client.save_to_json_file(full_path)

    best_params, values = ax_client.get_best_parameters()

    print(f'Best parameters:')
    print(f'num_states:\t {best_params["num_states"]}')
    print(f'kappa_sqrt:\t {best_params["kappa_sqrt"]:.2f}')

    print(f'Results stored at {full_path}.')
    print('Use the visualization_regression.ipynb notebook for viewing the stored experiment output.')
    print('Done')
else:
    print(f'Importing {__file__}')
