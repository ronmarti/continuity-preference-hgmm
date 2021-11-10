from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from models.utils.dataset_torch_preparator import split_for_validation

from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


class LogSoftmaxClassifier(LightningModule):
    """
    Classifier with single linear layer of weights before softmax applied.
    If `num_outputs` is ommitted, it is set to be equal to `num_inputs`.
    """
    def __init__(self, num_inputs, num_outputs=None, lr=0.1):
        super().__init__()
        self.lr = lr
        self.loss_function = nn.CrossEntropyLoss()
        if num_outputs is None:
            num_outputs = num_inputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        # self.weights_vec = torch.ones((1, self.num_inputs), dtype=torch.float)
        self.weights_vec = nn.Parameter(
            torch.ones((1, self.num_inputs), dtype=torch.float))
        self.bias = nn.Parameter(
            torch.zeros((1, self.num_inputs), dtype=torch.float)
            )
        # self.linear = nn.Linear(num_inputs, num_outputs)
        self.save_hyperparameters()

    def forward(self, input_vec):
        """
        Calculates cross-entropy-calibrated logsoftmax of input_vector.
        @param input_vec
        """
        # reweighted = self.weights_vec * (input_vec + self.bias)
        reweighted = F.relu(self.weights_vec) * (input_vec + self.bias)
        # reweighted = self.linear(input_vec)
        return F.log_softmax(reweighted, dim=1)
        # reweighted_corr = reweighted.exp() / reweighted.exp().sum(dim=1, keepdim=True)
        # return reweighted

    def training_step(self, batch, batch_idx):
        feature, label = batch
        log_probs = self(feature)
        loss = self.loss_function(log_probs, label)

        self.log('cross_entropy', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam([self.bias], lr=self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        feature, label = batch
        log_probs = self(feature)
        val_loss = self.loss_function(log_probs, label)

        self.log('val_cross_entropy', val_loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def init_patameters(self,
                        dset_tr: Dataset,
                        batch_size=1000):
        loader_train = DataLoader(dset_tr,
                                  batch_size=batch_size,
                                  shuffle=False)

        log_proba = torch.cat([pair[0] for pair in loader_train])
        max, _ = log_proba.max(dim=0, keepdim=True)
        min, _ = log_proba.min(dim=0, keepdim=True)
        total_min = min.min()
        self.weights_vec.data = total_min/min
        self.bias.data = -max

    def fit(self,
            dataset_train: Dataset,
            dataset_val: Dataset = None,
            max_epochs=100,
            lr=0.1,
            batch_size=1000,
            shuffle=True,
            balance_training_set=True,
            show_progress_bar=False,
            debug=False):
        """ Fits the multiclass classifier. Balancing not implemented.
        Arguments:
            dataset (TensorDataset): constructed from two tensors -
        input_features_tensor (N, num_inputs) of `torch.floats` and labels_tensor
        (N,) of `torch.long` oredered correspondingly.
        These represent inputs that are used to classify and labels
        are supposed to be values from 0 to self.num_outputs-1 of
        type long.
        """
        if dataset_val is None:
            dset_tr, dset_val = split_for_validation(dataset_train, ratio=0.4)
        else:
            dset_tr, dset_val = dataset_train, dataset_val
        # self.init_patameters(dset_tr, batch_size=batch_size)

        loader_train = DataLoader(dset_tr, batch_size=batch_size, shuffle=shuffle)
        loader_val = DataLoader(dset_val, batch_size=batch_size, shuffle=False)

        early_stop_callback = EarlyStopping(
            monitor='val_cross_entropy',
            min_delta=0.001,
            patience=100,
            verbose=True,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_cross_entropy',
            save_top_k=1,
            mode='min',
        )

        trainer = Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback,
                       early_stop_callback]
        )
        trainer.fit(self, loader_train, loader_val)
        return trainer.checkpoint_callback.best_model_path


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_features = torch.tensor([
        [-1.9050, -1.9252, -1.9179],
        [-1.9043, -1.9248, -1.9190],
        [-1.9048, -1.9241, -1.9204],
        [-1.9043, -1.9264, -1.9218],
        [-1.8631, -1.8007, -1.8168],
        [-1.8636, -1.8011, -1.8173],
        [-1.8613, -1.7994, -1.8180],
        [-1.8633, -1.7997, -1.8173],
        [-1.8535, -1.8130, -1.7999],
        [-1.8548, -1.8164, -1.8004],
        [-1.8534, -1.8154, -1.7998],
        [-1.8540, -1.8154, -1.7978]
    ]).float()

    plt.plot(data_features.detach())
    plt.show()
    data_labels = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    dataset = TensorDataset(data_features, data_labels)

    model1 = LogSoftmaxClassifier(3)
    model1.fit(dataset, dataset,
               max_epochs=500,
               show_progress_bar=True,
               debug=True,
               shuffle=False)
    # model1.init_patameters(dataset, batch_size=1000)

    # test_input = data_features[0, :].unsqueeze(0)
    # probs_test = model1(test_input).exp()
    # print(f'Features {test_input} is evaluated ' +
    #       f'by probabilities of classes as {probs_test}.')

    probs_test_all = model1(data_features).exp()
    plt.plot(probs_test_all.detach())
    plt.show()
    print(f'Model parameters:{[par for par in model1.named_parameters()]}')

else:
    print('Importing module softmax_classifier.py')
