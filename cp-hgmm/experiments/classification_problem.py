from torch import nn
from models.timeseries_classifier import TimeSeriesClassifier
from experiments.regression_problem import RegressionProblem
from typing import Dict, Iterable, Tuple
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Dataset
from models.utils.dataset_torch_preparator import DatasetContainer, convert_dset_to_tensor
from models.lit_system_em import LitEMTorch
from sklearn.metrics import accuracy_score
from datetime import datetime
from pathlib import Path


class ClassificationProblem(RegressionProblem):
    """
    Representation of the problem to be solved by optimization. Contains data
    and specification of metrics on that data.
    Classification problem searches for the best hyperparameters to maximize
    classification accuracy.
    """
    def __init__(self,
                 dset_container: DatasetContainer,
                 sample_length, num_outputs,
                 batch_size=100,
                 max_epochs=100,
                 n_trials=3,
                 val_ratio=0.4,
                 num_inner_retrials=3,
                 verbose=False,
                 logging_root=Path(f'.out/classifiers/')):
        super().__init__(dset_container,
                         sample_length, num_outputs,
                         batch_size,
                         max_epochs,
                         n_trials,
                         k_folds=1,
                         verbose=verbose)
        self.num_inner_retrials = num_inner_retrials
        self.val_ratio = val_ratio
        
        now_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.logging_root = logging_root / Path(f'{dset_container.name}_{now_str}')

    @property
    def y_test(self):
        y_test = self.dset_container.label_encoder.inverse_transform(
            convert_dset_to_tensor(self.dset_container.dataset_test)[2]
        )
        return y_test

    def utility_function(self, params_dict, *args):
        """
        This function needs to accept a set of parameter values as a dictionary.
        It should produce a dictionary of metric names to tuples of mean and
        standard error for those metrics.
        @param params_dict: dictionary of named parameters.
        `num_states`, `kappa_sqrt`.
        @return dict: {"metric_name": (metric_mean, metric_std)}, models_track
        """
        accuracies = []
        crossentropy_losses = []
        num_states = params_dict['num_states']
        kappa_sqrt = params_dict['kappa_sqrt']
        kappa = kappa_sqrt**2
        if kappa_sqrt < 0:
            kappa = -kappa
        models_track = []
        for i in range(self.n_trials):

            cf = TimeSeriesClassifier(
                max_epochs=self.max_epochs,
                batch_size=self.batch_size
            )
            cf.fit(self.dset_container,
                   n_trials=self.num_inner_retrials,
                   num_states=num_states,
                   kappa=kappa,
                   val_ratio=self.val_ratio)

            y_pred = cf.predict(self.dset_container.dataset_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            accuracies.append(accuracy)

            xentropy = self.crossentropy(cf, self.dset_container.dataset_test)
            crossentropy_losses.append(xentropy)

            metadata = {
                'test_accuracy': accuracy,
                'crossentropy_loss': xentropy.item(),
                'kappa_sqrt': params_dict['kappa_sqrt'],
                'num_states': num_states,
                'n_trials': self.num_inner_retrials
            }

            cf_path = cf.save_classifier(
                self.logging_root,
                metadata=metadata,
                classfier_name=f'{self.dset_container.name}_trial{i}'
            )
            models_track.append((accuracy, str(cf_path)))

        accuracies_tensor = torch.tensor(accuracies)
        crossentropy_tensor = torch.tensor(crossentropy_losses)

        return {
            'accuracy': (accuracies_tensor.mean(), accuracies_tensor.std()),
            'crossentropy_loss': (crossentropy_tensor.mean(), crossentropy_tensor.std())
        }, {'models_track': models_track}

    def crossentropy(self, cf: TimeSeriesClassifier, dset: Dataset):
        y_logprobs, y_true_labels = cf.predict_log_proba(dset)
        loss_fn = nn.CrossEntropyLoss()
        crossentropy_loss = loss_fn(y_logprobs, y_true_labels)
        return crossentropy_loss


if __name__ == "__main__":
    print(f'Running {__file__}')
    from pathlib import Path

    dataset_name = 'KR5'
    train_load_path = Path(r'N:\Datasets\TimeSeries\classification\IndustrialRobots\RobotKR5Consumption\RobotKR5Consumption_TRAIN.ts')
    # train_load_path = Path(r'D:\DATA_FAST\RobotKR5Consumption\RobotKR5Consumption_TRAIN.ts')
    test_load_path = None

    dset_container = DatasetContainer(dataset_name,
                                      mode='ts',
                                      test_ratio=0.98,
                                      dtype=torch.double,
                                      train=train_load_path,
                                      test=test_load_path)

    # dataset_name = 'SonyAIBORobotSurface1'
    # dset_container = DatasetContainer(dataset_name, mode='ucr')
    dset_container.plot_dataset()

    problem = ClassificationProblem(
        dset_container,
        sample_length=dset_container.data_len,
        num_outputs=dset_container.output_dim,
        max_epochs=200,
        n_trials=3,
        num_inner_retrials=5,
        verbose=False
    )
    parameters = {'num_states': 5,
                  'kappa_sqrt': 0}
    result, metadata = problem.utility_function(parameters)
    print(result)
    print('Done')

else:
    print(f'Importing {__file__}')
