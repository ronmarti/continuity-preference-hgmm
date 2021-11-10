from torch.functional import Tensor
from models.utils import plotting
from typing import Dict, Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
from torch.utils.data import random_split
from sktime.datasets import load_UCR_UEA_dataset
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.dataset import ConcatDataset


class DatasetContainer():
    '''Class that takes care of proper loading or downloading benchmark datasets
    of UCR timeseries dataset for classification task given their name.
    Specify datatype of timeseries, default is `torch.double`.
    If mode=="ts", paths to the training set and path to the test set must be provided in the **kwargs as
    "train": train_path, "test": test_path.
    @param test_ratio: if `None`, loads split train-test as predefined in dataset,
    otherwise if no test set is defined, it is extracted from
    the training set so that `train_ratio` is held out for training.
    '''
    def __init__(self, name, mode='ucr', test_ratio=None, dtype=torch.double, **kwargs):
        self.name = name
        self.dtype = dtype
        decision_map = {
            'ucr': self._load_ucr,
            'ts': self._load_ts
        }
        if mode not in decision_map:
            raise ValueError(f'Mode [{mode}] is not in decision map.')
        self.data_len = 0
        self.output_dim = 1
        self.test_ratio = test_ratio
        decision_map[mode](name, **kwargs)

    # @property
    # def data_len(self):
    #     # return self.dataset_train.tensors[0].size(1)
    #     return self.data_len

    # @property
    # def output_dim(self):
    #     return self.dataset_train.tensors[1].size(2)

    @property
    def unique_labels(self):
        return self.label_encoder.classes_

    def _convert_to_dataset(self, X, y, class_label):
        y_tensor = self._convert_pandas_to_torch(y)
        if X is None:
            length = y_tensor.size(1)
            num_batches = y_tensor.size(0)
            X_tensor = torch.linspace(0, length - 1, length).repeat(num_batches, 1)
        else:
            X_tensor = self._convert_pandas_to_torch(X)
        label_long_tensor = torch.tensor(
            self.label_encoder.transform(class_label),
            dtype=torch.long
        )
        return TensorDataset(X_tensor, y_tensor, label_long_tensor)

    def _load_ucr(self, name, **kwargs):
        if self.test_ratio is None:
            y_train, labels_train = load_UCR_UEA_dataset(name, split='train', return_X_y=True)
            y_test, labels_test = load_UCR_UEA_dataset(name, split='test', return_X_y=True)
        else:
            y, labels = load_UCR_UEA_dataset(name=name, return_X_y=True)
            strat_splitter = StratifiedShuffleSplit(n_splits=1, train_size=1-self.test_ratio)
            for tr_idx, test_idx in strat_splitter.split(y, labels):
                y_train, y_test, labels_train, labels_test = y.iloc[tr_idx], y.iloc[test_idx], labels.iloc[tr_idx], labels.iloc[test_idx]
                # labels, counts = np.unique(y_train, return_counts=True)
                # print(labels, counts)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels_train)

        self.dataset_train = self._convert_to_dataset(None, y_train, labels_train)
        self.data_len = self.dataset_train.tensors[0].size(1)
        self.output_dim = self.dataset_train.tensors[1].size(2)
        self.dataset_test = self._convert_to_dataset(None, y_test, labels_test)

    def _load_ts(self, name, train, test=None):
        '''Name is just decorative here. `train` == train set path, `test` == test set path.'''
        y_train, labels_train = load_from_tsfile_to_dataframe(train)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels_train)

        self.dataset_train = self._convert_to_dataset(None, y_train, labels_train)
        self.data_len = self.dataset_train.tensors[0].size(1)
        self.output_dim = self.dataset_train.tensors[1].size(2)
        if test is not None:
            y_test, labels_test = load_from_tsfile_to_dataframe(test)
            self.dataset_test = self._convert_to_dataset(None, y_test, labels_test)
        else:
            self.dataset_train, self.dataset_test = self.split_per_class_balanced_dataset(self.dataset_train, tr_val_ratio=self.test_ratio)

    def split_per_class_balanced_dataset(self, dataset: Dataset, tr_val_ratio=0.4):
        """
        Cuts-out portion ov tr_val_ratio of the given dataset from each class.
        This results int having validation set with approx. the same class ratios.
        """
        tr_sets = []
        val_sets = []
        for label in self.unique_labels:
            Xs, ys, labels = self._select_class(dataset, label)
            # encoded_label = self.label_encoder.transform([label])
            dset = TensorDataset(Xs, ys, labels)
            tr_set, val_set = self._split_for_validation(
                dset,
                ratio=tr_val_ratio
                )
            tr_sets.append(tr_set)
            val_sets.append(val_set)
        return ConcatDataset(tr_sets), ConcatDataset(val_sets)

    def class_train_datasets_generator(self):
        """
        Function to be used for looping over classes contained in training dataset in the container.
        """
        labels = self.unique_labels
        for label in labels:
            yield *self.select_class_train(label), label

    def split_stratified_to_triplets(self, dataset: Dataset, tr_val_ratio=0.4, val_a_b_ratio=0.5) -> dict:
        """
        @return per_class_dsets_splitted: dictionary of {label: (tr_set, val_set_a, val_set_b)}
        """
        per_class_dsets_splitted = {}
        for label in self.unique_labels:
            dset = TensorDataset(*self._select_class(dataset, label))
            tr_set, val_set = self._split_for_validation(
                dset,
                ratio=tr_val_ratio
                )
            val_set_a, val_set_b = self._split_for_validation(
                val_set,
                ratio=val_a_b_ratio)
            per_class_dsets_splitted[label] = (tr_set, val_set_a, val_set_b)
        return per_class_dsets_splitted

    def split_stratified(self,
                         dataset: Dataset,
                         val_ratio=0.4
                         ) -> Dict[str, Tuple[Dataset, Dataset]]:
        """
        @param dataset: dataseet to split into train-validation subsets.
        @param val_ratio: validation size ratio to the whole dataset size.
        @return per_class_dsets_splitted: stratified train-validation splitted
        datasets in dictionary {model_label: (tr_set, val_set)}.
        """
        dsets_splitted = {}
        for label in self.unique_labels:
            dset = TensorDataset(*self._select_class(dataset, label))
            tr_set, val_set = self._split_for_validation(dset, val_ratio)
            dsets_splitted[label] = (tr_set, val_set)
        return dsets_splitted

    def select_class_train(self, label, is_label_raw=True):
        """
        Selects only occurences of selected class label from the training set.
        @param label: class label to select.
        @is_label_raw: raw label is what is stored in the original dataset,
        plain label is `long` in range [0, num_classes-1].
        @return X, y: tuple of two Dataset of timestamps and values respectively.
        """
        return self._select_class(self.dataset_train, label, is_label_raw)

    def select_class_test(self, label, is_label_raw=True):
        """
        Selects only occurences of selected class label from the test set.
        @param label: class label to select.
        @is_label_raw: raw label is what is stored in the original dataset,
        plain label is `long` in range [0, num_classes-1].
        @return X, y: tuple of two Dataset of timestamps and values respectively.
        """
        assert self.dataset_test is not None, 'The dataset_test is None, cannot select from it.' 
        return self._select_class(self.dataset_test, label, is_label_raw)

    def _select_class(self, dataset: TensorDataset, label, is_label_raw=True):
        '''`dataset` is TensorDataset.'''
        if is_label_raw:
            label = self.label_encoder.transform([label])[0]
        selected_series = [
            series
            for X, series, label_id in dataset if label_id == label
        ]
        selected_Xs = [
            X
            for X, series, label_id in dataset if label_id == label
        ]
        selected_labels = [
            label_id
            for X, series, label_id in dataset if label_id == label
        ]
        return (torch.stack(selected_Xs, dim=0),
                torch.stack(selected_series, dim=0),
                torch.stack(selected_labels, dim=0))

    def select_from_dataset(self,
                       dset: Dataset,
                       keys_iterable: Iterable,
                       val_ratio=0.4) -> Tuple[Dataset, Dataset]:
        """ Primary way of selecting class examples by their names.
        Selects given labels from given dataset and returns aggregated train-validation pair.
        """
        dsets_splitted = self.split_stratified(dset,
                                               val_ratio)
        pairs = [dsets_splitted[key] for key in keys_iterable]

        tr_set = ConcatDataset([tr for tr, _ in pairs])
        val_set = ConcatDataset([val for _, val in pairs])
        return tr_set, val_set

    def _convert_pandas_to_torch(self, X):
        X_np = from_nested_to_3d_numpy(X)
        dataset = torch.tensor(np.moveaxis(X_np, 1, -1), dtype=self.dtype).unsqueeze(-1)
        return dataset

    def get_train_val_split(self, ratio=0.2):
        '''Returns random split of the stored `dataset_train` into `train_set`, `val_set`, where
        `ratio` is ratio of validation data out of the whole training dataset.'''
        train_set, val_set = self._split_for_validation(self.dataset_train)
        return train_set, val_set

    def _split_for_validation(self, y: Dataset, ratio=.2):
        '''Handler for data split into `train_set` and `val_set` according to
        given `ratio` which is validation size ratio to the whole dataset y size.'''
        data_len = len(y)

        assert data_len >= 2, f'Expected at least 2 training samples, {data_len} provided.'
        train_len = round((1-ratio) * data_len)
        train_len = max(train_len, 1)  # at least 1 training example
        val_len = data_len - train_len
        if val_len == 0:
            val_len = 1
            train_len = data_len - val_len
        train_set, val_set = random_split(y, [train_len, val_len])
        return train_set, val_set

    def plot_dataset(self, with_title=False, show=True, xlabel=None, ylabel=None):
        fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))
        labels_numeric = self.label_encoder.transform(self.label_encoder.classes_)
        for label_numeric in labels_numeric:
            label = self.label_encoder.inverse_transform([label_numeric])[0]
            x, y, lbl = self.select_class_train(
                label_numeric,
                False)
            mean = y.squeeze(-1)[:, :, 0].mean(dim=0)
            std = y.squeeze(-1)[:, :, 0].std(dim=0)
            time = x.mean(dim=0)
            ax.plot(time, mean, label=f'{label} $\pm \sigma$', LineWidth=2)
            ax.fill_between(time, mean-std, mean+std,
                             alpha=0.5)
        
        ax.set_xlim((x.min(), x.max()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend(loc='upper right')
        if with_title:
            ax.title(f'Classes in Dataset {self.name}')
        plt.tight_layout()
        if show:
            plt.show()

    def plot_given_dataset(self, dset: Dataset, absolute_range=False):
        for batch in DataLoader(dset, batch_size=len(dset)):
            x = batch[0]
            y = batch[1]
            for dim in range(y.size(2)):
                mean = y.squeeze(-1)[:, :, dim].mean(dim=0)
                std = y.squeeze(-1)[:, :, dim].std(dim=0)
                time = x.mean(dim=0)
                plt.plot(time, mean, label=f'$\mu_{dim}$')
                if absolute_range:
                    minimum, _ = y.squeeze(-1)[:, :, dim].min(dim=0)
                    maximum, _ = y.squeeze(-1)[:, :, dim].max(dim=0)
                    plt.fill_between(time, minimum, maximum,
                                    alpha=0.5, label=f'$\pm\sigma_{dim}$')
                else:
                    plt.fill_between(time, mean-std, mean+std,
                                    alpha=0.5, label=f'$\pm\sigma_{dim}$')
            plt.legend()
            plt.title(f'Dataset {self.name}')
        plt.show()

    def get_kfold_splits(self, dset: Dataset, k: int) -> Iterable[Tuple[Dataset, Dataset]]:
        """
        Prepares `k` datasets splits that are exclusive k-folds of training data and
        validation data. Iterate through `k` pairs (tr, val) k-times to do the
        k-fold cross-validation.

        If `k` is 1, entire dataset is returned as training set and an empty
        dataset is returned as validation dataset.
        """
        length = len(dset)
        num_elements = length // k
        if num_elements <= 1:
            num_elements = 1
            k = length
        sizes = (k-1)*[num_elements]
        sizes.append(length-(k-1)*num_elements)
        kfolds = set(random_split(dset, sizes))
        if k == 1:
            rollout = [(kfolds, Dataset())]
        else:
            rollout = [(ConcatDataset(kfolds - {kth}), kth) for kth in kfolds]
        return rollout

    def info(self):
        return {
            'data_len': self.data_len,
            'output_dim': self.output_dim,
            'num_classes': len(self.unique_labels),
            'num_samples_test': len(self.dataset_test),
            'num_samples_train': len(self.dataset_train),
            'name': self.name
        }


def convert_dset_to_tensor(dset: Dataset,
                           batch_size: int = 100) -> Iterable[Tensor]:
    """Iterates through dataset and preps tuple of tensors from it.
    """
    ldr = DataLoader(dset,
                     batch_size=batch_size,
                     shuffle=False)
    tensors = [torch.cat(list_tensors, dim=0) for list_tensors in zip(*ldr)]
    return tensors


def split_for_validation(dataset: Dataset, ratio=.33) -> Tuple[Dataset, Dataset]:
    '''Handler for data split into `train_set` and `val_set` according to
    given `ratio` which is validation size ratio to the whole dataset y size.
    @return (train_set, val_set): Datasets'''
    # y = self.trim_length(y)
    data_len = len(dataset)
    assert data_len >= 2, f'Expected at least 2 training samples, {data_len} provided.'
    train_len = round((1-ratio) * data_len)
    train_len = max(train_len, 1)  # at least 1 training example
    val_len = data_len - train_len
    if val_len == 0:
        val_len = 1
        train_len = data_len - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    return train_set, val_set


if __name__ == "__main__":
    from datetime import datetime
    now_str = datetime.now().strftime('%Y%m%dT%H%M%S')
    # dset_container = DatasetContainer('SonyAIBORobotSurface2',
    #                                      mode='ucr',
    #                                      dtype=torch.float)
    # # dset_container.plot_dataset()

    # strat_dset = dset_container.split_stratified(dset_container.dataset_train)
    # plotting.plot_stratified_dset(strat_dset)

    # res = dset_container.get_kfold_splits(dset_container.dataset_train, 7)
    # for tr, val in res:
    #     dset_container.plot_given_dataset(val)
    #     dset_container.plot_given_dataset(tr)

    dataset_name = 'kr210'
    xlabel = 'Sample $t$'
    ylabel = 'Power [kW]'
    train_load_path = Path('data_for_tests/RobotMBConsumption/RobotMBConsumption_ALL.ts')
    test_load_path = None

    dset_container = DatasetContainer(dataset_name,
                                      mode='ts',
                                      test_ratio=0.01,
                                      dtype=torch.double,
                                      train=train_load_path,
                                      test=test_load_path)
    dset_container.plot_dataset(with_title=False,
                                show=False,
                                xlabel=xlabel,
                                ylabel=ylabel)
    out_root = Path('.out/illustrations')
    out_root.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'.out/illustrations/plot_dataset_{dataset_name}.pdf')
    plt.show()

    dataset_name = 'SonyAIBORobotSurface1'
    xlabel = 'Sample $t$'
    ylabel = 'X-acceleration [-]'
    dset_container = DatasetContainer(dataset_name, mode='ucr', test_ratio=0.01)

    dset_container.plot_dataset(with_title=False,
                                show=False,
                                xlabel=xlabel,
                                ylabel=ylabel)
    plt.savefig(f'.out/illustrations/plot_dataset_{dataset_name}.pdf')

    print('End of tests.')

else:
    print(f'Importing dataset_torch_preparator.py')
