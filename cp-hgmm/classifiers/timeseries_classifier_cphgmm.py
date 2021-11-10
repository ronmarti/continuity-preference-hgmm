from os import PathLike
from typing import Dict, Tuple
from models.lit_system_em import LitEMTorch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Dataset

from models.softmax_classifier import LogSoftmaxClassifier
from sklearn.metrics import log_loss
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from models.utils.dataset_torch_preparator import DatasetContainer, convert_dset_to_tensor
import json


class TimeSeriesClassifier():
    '''
    Complies partially with `sklearn` classifier interface.
    Classifier of timeseries of the uniform lengths with output calibration by softmax.
    @param trained_models_paths_dict: if not None, the models of particular classes
    are loaded from checkpoints on given paths.
    '''

    def __init__(self,
                 max_epochs=100,
                 batch_size=100,
                 verbose=True):

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.label_encoder = None
        self.models: Dict[str, LitEMTorch] = None
        self.length = None
        self.num_outputs = None
        self.verbose = verbose

        self.models_results = pd.DataFrame(
            columns=['pos_label',
                     'trial_id',
                     'checkpoint_path',
                     'pos_tr_elpd',
                     'pos_val_elpd',
                     'neg_elpd',
                     'gap_elpd',
                     'gap_pos_tr_val_elpd'])

    @property
    def unique_labels(self):
        return self.label_encoder.classes_

    @property
    def ordered_labels(self):
        numerics_ordered = np.arange(len(self.label_encoder.classes_))
        ordered_lbls = self.label_encoder.inverse_transform(numerics_ordered)
        return ordered_lbls

    @staticmethod
    def load_classifier(models_paths_file: PathLike):
        root = Path(Path(models_paths_file).root)
        with open(models_paths_file) as f:
            models_dict = json.load(f)
        models_paths = {
            lbl: root / Path(mdl_filename)
            for lbl, mdl_filename in models_dict['models_dict'].items()
        }
        classifier = TimeSeriesClassifier()
        classifier.models = classifier.load_models(models_paths)
        return classifier

    def save_classifier(self, root: PathLike,
                        metadata=None,
                        classfier_name: str = None):
        Path.mkdir(root, parents=True, exist_ok=True)
        mdl_paths = [(lbl, Path(root) / Path(f'{lbl}_LitEMTorch.pt'), mdl)
                     for lbl, mdl in self.models.items()]
        for lbl, path, mdl in mdl_paths:
            torch.save(mdl.state_dict(), path)
        models_dict = {lbl: str(path.name) for lbl, path, _ in mdl_paths}
        if classfier_name is None:
            classfier_name = ''
        data_dict_to_save = {
            'models_dict': models_dict,
            'metadata': metadata,
            'training_log': self.models_results.to_dict()
        }
        cf_path = Path(root) / Path(f'{classfier_name}_cf_mdls_paths.json')
        with open(cf_path, 'w', encoding ='utf8') as json_file:
            json.dump(data_dict_to_save, json_file)
        return cf_path


    def load_models(self, models_dict: Dict[str, PathLike]) -> Dict[str, LitEMTorch]:
        """
        Loads models based on dictionary of {class_name: checkpoint_path}.
        """
        models = {}
        for label, path in models_dict.items():
            models[label] = LitEMTorch.load_from_checkpoint(Path(path))
        return models

    def fit(self,
            data_container: DatasetContainer,
            n_trials=1,
            num_states=2,
            kappa=0.,
            val_ratio=0.4
            ):
        """
        @param n_trials: number of random restarts to try to find the best performing
        model per class.
        @param kappa: continuity preference parameter. It's the inverse of variance
        of time-adjacent evolution matrix' elements.
        """
        self.label_encoder = data_container.label_encoder
        self.length = data_container.data_len
        self.num_outputs = data_container.output_dim
        dsets_stratified_trval = data_container.split_stratified(
            data_container.dataset_train,
            val_ratio=val_ratio
        )
        self.fit_candidate_models(dsets_stratified_trval,
                              self.num_outputs,
                              num_states,
                              kappa,
                              n_trials)

        self.select_best_models()
        self.fit_calib_layer(dsets_stratified_trval)

    def fit_candidate_models(self,
            stratified_dset:  Dict[str, Tuple[Dataset, Dataset]],
            num_outputs,
            num_states=2,
            kappa=0.,
            n_trials=1
            ):
        """
        Fit classifier from data_container data.
        @param n_trials: number of random restarts per model to search for
            the best-fitting one
        @return self: object
        """
        starting_state_dicts = []
        for trial_id in range(n_trials):
            model = LitEMTorch(self.length,
                               num_states=num_states,
                               num_outputs=num_outputs,
                               kappa=kappa)
            starting_state_dicts.append(model.state_dict())

        labels_set = set(self.unique_labels)
        for pos_label in labels_set:
            neg_labels = labels_set - {pos_label}

            pos_tr_set, pos_val_set = stratified_dset[pos_label]
            neg_tr_set = ConcatDataset(
                [stratified_dset[lbl][0] for lbl in neg_labels]
            )

            pos_tr_ldr = DataLoader(pos_tr_set, batch_size=self.batch_size)
            pos_val_ldr = DataLoader(pos_val_set, batch_size=self.batch_size)
            neg_ldr = DataLoader(neg_tr_set, batch_size=self.batch_size)

            for trial_id, init_state_dict in enumerate(starting_state_dicts):
                model = LitEMTorch(self.length,
                                   num_states=num_states,
                                   num_outputs=num_outputs,
                                   kappa=kappa)
                model.load_state_dict(init_state_dict)
                chckpt_path = model.fit(pos_tr_ldr,
                                        pos_val_ldr,
                                        max_epochs=self.max_epochs,
                                        verbose=self.verbose)
                try:
                    pos_tr_elpd = model.eval_elpd_ldr(pos_tr_ldr).item()
                    pos_val_elpd = model.eval_elpd_ldr(pos_val_ldr).item()
                    neg_elpd = model.eval_elpd_ldr(neg_ldr).item()

                    self.models_results = self.models_results.append(
                        {'pos_label': pos_label,
                        'trial_id': trial_id,
                        'checkpoint_path': chckpt_path,
                        'pos_tr_elpd': pos_tr_elpd,
                        'pos_val_elpd': pos_val_elpd,
                        'neg_elpd': neg_elpd,
                        'gap_elpd': pos_val_elpd - neg_elpd,
                        'gap_pos_tr_val_elpd': pos_tr_elpd - pos_val_elpd},
                        ignore_index=True
                    )
                except RuntimeError as err:
                    print(f'Throwing err {err}')

    def select_best_models(self):
        """ Sarches `self.models_results` for the best models for each unique
        label. Resulting ensemble is stored in dictionary self.models.
        """
        selected_models = {}
        mdl_results = self.models_results
        mdl_lbls = mdl_results['pos_label'].unique()
        for mdl_lbl in mdl_lbls:
            trials = mdl_results[mdl_results['pos_label'] == mdl_lbl]
            best_mdl_row = trials.iloc[trials['gap_elpd'].argmax()]
            best_mdl_path = best_mdl_row['checkpoint_path']
            selected_models[mdl_lbl] = best_mdl_path
        self.models = self.load_models(selected_models)

    def fit_calib_layer(self,
                        stratified_dset:  Dict[str, Tuple[Dataset, Dataset]]):
        """
        @param stratified_dset: dictionary of stratified train-validation splitted
        datasets in dictionary {model_label: (tr_set, val_set)}.
        Use DatasetContainer.split_stratified to get such dictionary.
        """
        # generate features for classification
        train_feat = []
        val_feat = []
        for label, (tr_set, val_set) in stratified_dset.items():
            feature_dset_tr = self.calculate_feature_vectors(
                tr_set,
                self.ordered_labels
            )
            train_feat.append(feature_dset_tr)

            feature_dset_val = self.calculate_feature_vectors(
                val_set,
                self.ordered_labels
            )
            val_feat.append(feature_dset_val)

        # join dataset to somewhat obey the sklearn interface
        dset_log_proba_tr = ConcatDataset(train_feat)
        dset_log_proba_val = ConcatDataset(val_feat)

        num_classes = len(self.label_encoder.classes_)
        self._fit_calib_layer(dset_log_proba_tr,
                          dset_log_proba_val,
                          num_classes)

    def calculate_feature_vectors(self, dset: Dataset, ordered_lbls: list):
        """
        Transforms timeseries into vector of scores given by each model.
        @param dset: torch.utils.data.Dataset of tensors of size (b, length, num_features, 1)
        to transform to feature tensor.
        @return dataset: ConcatDataset of batch tensor of log-probs per model ordered as in 
        `self.models`, size (b_len, n_models) and its true_label repeated of size
        (b_len).
        """
        assert isinstance(dset, Dataset), f'Given dset is not an instance inheritted from torch Dataset.'
        loader = DataLoader(dset, batch_size=self.batch_size, shuffle=False)
        transformed_batches = []
        for batch in loader:
            log_probs_per_model = []
            for lbl in ordered_lbls:
                model = self.models[lbl]
                X_log_proba_tensor = model.predict_log_proba(batch[1]).mean(dim=1).float()
                log_probs_per_model.append(X_log_proba_tensor)
            stacked_x = torch.stack(log_probs_per_model, dim=1).detach()
            transformed_batches.append(TensorDataset(stacked_x, batch[2]))
        catted_transformed_dataset = ConcatDataset(transformed_batches)
        return catted_transformed_dataset

    def _fit_calib_layer(
            self,
            dset_log_proba_train,
            dset_log_proba_val,
            num_classes,
            max_epochs=10000,
            batch_size=1000):
        """
        num_features typically equals to num_classes.
        @param dset_log_proba_train: Dataset of tensors (feature_tensor, label_tensor), where
        feature_tensor is size (b, n_classes) and label_tensor is size (b).
        """
        self.calibarion_layer = LogSoftmaxClassifier(num_classes)
        self.calibarion_layer.fit(
            dset_log_proba_train,
            dset_log_proba_val,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=0.01,
            show_progress_bar=True,
            debug=True
        )

    def predict_log_proba(self, dset: Dataset):
        '''Iterates over all models and produces log-proba per model - stacked into tensor (n_features).
        @param dset
        @return stack: tensor (b, num_features) of log-probs (log-likelihoods) of each model.
        '''
        feature_vectors_dset = self.calculate_feature_vectors(
            dset,
            self.ordered_labels
        )

        tensors = convert_dset_to_tensor(feature_vectors_dset)

        return tensors[0], tensors[1]

    def predict_proba(self, dset: Dataset):
        X_log_proba, lbls = self.predict_log_proba(dset)
        logsoftmax = self.calibarion_layer(X_log_proba)
        return logsoftmax.exp().detach().numpy()


    def predict(self, dset: Dataset):
        """
        Predicts classes of given samples in the form of feature tensors.
        @param dset: input dataset containing feature tensor of shape (n_samples, n_features)
        @return classification: array of encoded class labels in the same order
        as dset is ordered. (n_samples,)
        """
        X_log_proba, lbls = self.predict_log_proba(dset)
        logsoftmax = self.calibarion_layer(X_log_proba)
        maxima, argmaxima = logsoftmax.max(dim=1)
        class_labels = self.ordered_labels[argmaxima.detach().numpy()]
        return class_labels


if __name__ == "__main__":
    
    # BasicMotions, ArrowHead, Haptics, SonyAIBORobotSurface2, Epilepsy

    # torch.manual_seed(53)

    dataset_name = 'SonyAIBORobotSurface1'
    dset_container = DatasetContainer(dataset_name, mode='ucr', test_ratio=0.99)

    # dataset_name = 'KR2700'
    # train_load_path = Path('data_for_tests/RobotMBConsumption/RobotMBConsumption_ALL.ts')
    # test_load_path = None
    # dset_container = DatasetContainer(dataset_name,
    #                                   mode='ts',
    #                                   test_ratio=0.80,
    #                                   dtype=torch.double,
    #                                   train=train_load_path,
    #                                   test=test_load_path)
    # dset_container.plot_dataset()

    length = dset_container.data_len
    num_states = 5
    num_outputs = dset_container.output_dim
    kappa = 69**2
    max_epochs = 200
    batch_size = 1000
    n_trials = 5
    val_ratio = 0.5

    now_str = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_root = Path(f'.out/classifiers/{dataset_name}_{now_str}')

    cf = TimeSeriesClassifier(
        max_epochs=max_epochs,
        batch_size=batch_size
    )
    cf.fit(dset_container,
           n_trials=n_trials,
           num_states=num_states,
           kappa=kappa,
           val_ratio=val_ratio)

    y_pred = cf.predict(dset_container.dataset_test)

    test_data = convert_dset_to_tensor(dset_container.dataset_test)
    y_test = dset_container.label_encoder.inverse_transform(test_data[2])

    y_pred_proba = cf.predict_proba(dset_container.dataset_test)
    xentropy = log_loss(y_test, y_pred_proba)
    print(f'Crossentropy: {xentropy:.3f}')

    metadata = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'crossentropy_loss': xentropy,
        'kappa': kappa,
        'num_states': num_states,
        'n_trials': n_trials
    }
    cf.save_classifier(out_root,
                       metadata=metadata,
                       classfier_name=dataset_name)

    cm_raw = confusion_matrix(
        y_test, y_pred,
        labels=cf.unique_labels
    )
    cm_normalized = confusion_matrix(
        y_test, y_pred,
        labels=cf.unique_labels,
        normalize='true'
    )


    fig, axes = plt.subplots(2, figsize=(6, 8))

    cm_raw_display = ConfusionMatrixDisplay(
        cm_raw,
        display_labels=cf.unique_labels
    ).plot(ax=axes[0], cmap=plt.cm.Blues)
    cm_norm_display = ConfusionMatrixDisplay(
        cm_normalized,
        display_labels=cf.unique_labels
    ).plot(ax=axes[1], cmap=plt.cm.Blues)
    plt.tight_layout()
    print(f'Accuracy {accuracy_score(y_test, y_pred)}')
    plt.show()

    print('End of tests.')

else:
    print(f'Importing basic_classifier.py')
