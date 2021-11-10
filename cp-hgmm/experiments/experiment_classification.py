from experiments.classification_problem import ClassificationProblem
from typing import Iterable
import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data import TensorDataset

from ax.service.managed_loop import optimize
from ax.storage.json_store.save import save_experiment
from ax.service.ax_client import AxClient

from models.utils.dataset_torch_preparator import DatasetContainer
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

    dataset_name = 'KR2700'
    train_load_path = Path('data_for_tests/RobotMBConsumption/RobotMBConsumption_ALL.ts')
    test_load_path = None
    # train_load_path = Path('data_for_tests/RobotMBConsumption/RobotMBConsumption_TRAIN.ts')
    # test_load_path = Path('data_for_tests/RobotMBConsumption/RobotMBConsumption_TEST.ts')
    # train_load_path = Path('N:/Datasets/TimeSeries/classification/IndustrialRobots/RobotMBConsumption/RobotMBConsumption_TRAIN.ts')
    # test_load_path = Path('N:/Datasets/TimeSeries/classification/IndustrialRobots/RobotMBConsumption/RobotMBConsumption_TEST.ts')

    # dataset_name = 'KR5'
    # train_load_path = Path(r'N:\Datasets\TimeSeries\classification\IndustrialRobots\RobotKR5Consumption\RobotKR5Consumption_TRAIN.ts')
    # # train_load_path = Path(r'D:\DATA_FAST\RobotKR5Consumption\RobotKR5Consumption_TRAIN.ts')
    test_load_path = None

    dset_container = DatasetContainer(dataset_name,
                                      mode='ts',
                                      test_ratio=0.8,
                                      dtype=torch.double,
                                      train=train_load_path,
                                      test=test_load_path)

    # ECG200, EOGHorizontalSignal
    # dataset_name = 'SonyAIBORobotSurface1'
    # dset_container = DatasetContainer(dataset_name, mode='ucr', test_ratio=0.99)
    dset_container.plot_dataset()

    now_str = datetime.now().strftime('%Y%m%dT%H%M%S')
    experiment_root = Path(f'.out/experiments/cls_{dataset_name}')
    experiment_root.mkdir(parents=True, exist_ok=True)
    filename = Path(f'{dataset_name}_{now_str}.json')

    problem = ClassificationProblem(
        dset_container,
        sample_length=dset_container.data_len,
        num_outputs=dset_container.output_dim,
        max_epochs=200,
        n_trials=3,
        val_ratio=0.5,
        num_inner_retrials=5,
        verbose=False,
        logging_root=experiment_root
    )

    ax_client = AxClient()
    ax_client.create_experiment(
        name=f'{dataset_name}_optimal_parameters_search',
        parameters=[
            {
                'name': 'num_states',
                'type': 'range',
                'bounds': [3, 7],
                'value_type': 'int'
            },
            {
                'name': 'kappa_sqrt',
                'type': 'range',
                'bounds': [0, 1000],
                'value_type': 'float'
            }
        ],
        objective_name='crossentropy_loss',
        minimize=True,
        parameter_constraints=[]
    )

    load_preceding_clients_experiments(ax_client, root=experiment_root)

    full_path = str(experiment_root / filename)

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
    print('Use the visualization_classification.ipynb notebook for viewing the stored experiment output.')
    print('Done')
else:
    print(f'Importing {__file__}')
