from typing import Dict, Iterable, Tuple
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Dataset
from models.utils.dataset_torch_preparator import DatasetContainer
from models.lit_system_em import LitEMTorch


class RegressionProblem():
    """
    Representation of the problem to be solved by optimization. Contains data
    and specification of metrics on that data.
    Regression problem searches for the best hyperparameters to fit the sampled
    function.
    """

    def __init__(self,
                 dset_container: DatasetContainer,
                 sample_length, num_outputs,
                 batch_size=100,
                 max_epochs=100,
                 n_trials=3,
                 k_folds=10,
                 verbose=False):
        self.dset_container = dset_container
        self.n_trials = n_trials
        self.sample_length = sample_length
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.k_folds = k_folds
        self.verbose = verbose

    @property
    def tr_set_y(self):
        return self.tr_set.tensors[1]

    @property
    def tr_set_x(self):
        return self.tr_set.tensors[0]

    @property
    def val_set_y(self):
        return self.val_set.tensors[1]

    @property
    def val_set_x(self):
        return self.val_set.tensors[0]

    @property
    def unique_labels(self) -> Tuple[str]:
        return self.dset_container.unique_labels

    def prep_data_kfold(self, label, k=10) -> Iterable[Tuple[Dataset, Dataset]]:
        dset = TensorDataset(*self.dset_container.select_class_train(label))
        return self.dset_container.get_kfold_splits(dset, k)

    def prep_data(self, keys_iterable: Iterable, tr_val_ratio=0.4):
        per_class_dsets_splitted = self.dset_container.split_stratified_to_triplets(
            self.dset_container.dataset_train,
            tr_val_ratio=tr_val_ratio
            )
        triplets = [per_class_dsets_splitted[key] for key in keys_iterable]

        tr_set = ConcatDataset([tr for tr, _, _ in triplets])
        val_set_a = ConcatDataset([val_a for _, val_a, _ in triplets])
        val_set_b = ConcatDataset([val_b for _, _, val_b in triplets])
        return tr_set, val_set_a, val_set_b

    def utility_function(self, params_dict, *args):
        """
        This function needs to accept a set of parameter values as a dictionary.
        It should produce a dictionary of metric names to tuples of mean and
        standard error for those metrics.
        @param params_dict: dictionary of named parameters.
        `num_states`, `kappa_sqrt`.
        @return dict: {"metric_name": (metric_mean, metric_std)
        """
        # n_trials = self.n_trials
        pos_tr_elpd_results = []
        pos_tr_val_elpd_dists = []
        pos_elpd_results = []
        neg_elpd_results = []
        lpds_diffs = []
        # num_epochs = []
        metadata = {}
        labels_set = set(self.unique_labels)
        for pos_label in labels_set:
            num_states = params_dict['num_states']
            kappa_sqrt = params_dict['kappa_sqrt']
            kappa = kappa_sqrt**2
            if kappa_sqrt < 0:
                kappa = -kappa
            neg_labels = labels_set - {pos_label}
            kfolds = self.prep_data_kfold(pos_label, k=self.k_folds)
            neg_tr_set, neg_val_a_set, neg_val_b_set = self.prep_data(neg_labels, tr_val_ratio=0.6)
            neg_val_ldr = DataLoader(neg_val_a_set, batch_size=self.batch_size)
            models_track = []
            for i in range(self.n_trials):
                for kth in kfolds:
                    regression = LitEMTorch(length=self.sample_length,
                                            num_states=num_states,
                                            num_outputs=self.num_outputs,
                                            kappa=kappa)

                    pos_tr_set, pos_val_a_set = kth
                    tr_ldr = DataLoader(pos_tr_set, batch_size=self.batch_size)
                    val_ldr = DataLoader(pos_val_a_set, batch_size=self.batch_size)

                    model_path = regression.fit(tr_ldr, val_ldr, max_epochs=self.max_epochs, verbose=self.verbose)
                    pos_tr_elpd_result = regression.eval_elpd_ldr(tr_ldr)
                    pos_elpd_result = regression.eval_elpd_ldr(val_ldr)
                    neg_elpd_result = regression.eval_elpd_ldr(neg_val_ldr)
                    # num_epochs.append(regression.current_epoch)
                    pos_tr_elpd_results.append(pos_tr_elpd_result)
                    pos_elpd_results.append(pos_elpd_result)
                    neg_elpd_results.append(neg_elpd_result)
                    pos_tr_val_elpd_dists.append(abs(pos_tr_elpd_result - pos_elpd_result))
                    lpds_diff = pos_elpd_result - neg_elpd_result
                    lpds_diffs.append(lpds_diff)
                    models_track.append((lpds_diff, model_path))
            metadata[pos_label] = models_track

        pos_lpds_tensor = torch.tensor(pos_elpd_results)
        pos_tr_elpd_tensor = torch.tensor(pos_tr_elpd_results)
        pos_tr_val_elpd_diff_tensor = torch.tensor(pos_tr_val_elpd_dists)
        neg_lpds_tensor = torch.tensor(neg_elpd_results)
        lpds_diff_tensor = torch.tensor(lpds_diffs)
        gap_elpds = lpds_diff_tensor.mean()
        gap_elpds_std = lpds_diff_tensor.std()
        gaps_mean = gap_elpds - pos_tr_val_elpd_diff_tensor.mean()
        gaps_std = torch.sqrt(gap_elpds_std**2 + pos_tr_val_elpd_diff_tensor.std()**2)
        return {
            'gap_elpd': (gap_elpds, gap_elpds_std),
            'pos_elpd': (pos_lpds_tensor.mean(), pos_lpds_tensor.std()),
            'neg_elpd': (neg_lpds_tensor.mean(), neg_lpds_tensor.std()),
            'pos_tr_elpd': (pos_tr_elpd_tensor.mean(), pos_tr_elpd_tensor.std()),
            'gap_pos_tr_val_elpd': (pos_tr_val_elpd_diff_tensor.mean(), pos_tr_val_elpd_diff_tensor.std()),
            'gaps': (gaps_mean, gaps_std)
            }, metadata


if __name__ == "__main__":
    print(f'Running {__file__}')
    from pathlib import Path

    # dataset_name = 'SonyAIBORobotSurface1'
    # dset_container = DatasetContainer(dataset_name, mode='ucr', test_ratio=0.99)

    # dataset_name = 'KR2700'
    # train_load_path = Path('data_for_tests/RobotMBConsumption/RobotMBConsumption_ALL.ts')

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

    dset_container.plot_dataset()

    problem = RegressionProblem(
        dset_container,
        sample_length=dset_container.data_len,
        num_outputs=dset_container.output_dim,
        max_epochs=100,
        n_trials=1,
        verbose=False
    )
    parameters = {
        'num_states': 5,
        'kappa_sqrt': -1
    }
    result, metadata = problem.utility_function(parameters)

    print(result)
    print(f'gap_elpd mean = {result["gap_elpd"][0].item()}')
    print(f'gap_var mean = {result["gap_pos_tr_val_elpd"][0].item()}')

    print('Done')

else:
    print(f'Importing {__file__}')
