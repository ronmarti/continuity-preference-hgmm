from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
import math


from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from models.smooth_kalman import SmoothKalman
from models.tv_hmm_torch import ModelHMMTorch

from models.utils.dataset_torch_preparator import DatasetContainer


class LitEMBase(LightningModule):
    """
    ML System for HGMM model.
    @param length: length of dataset.
    @param num_states: number of hidden states to use.
    @param num_outputs: number of outputs of the observable in the dataset.
    @param kappa: continuity preference strength - precision (inverse variance)
    of the parameter gaussian process.
    """

    def __init__(self):
        super().__init__()
        self.smoother = None
        self.dummy_param = torch.nn.Parameter(
            torch.rand([1, 1],
                       dtype=torch.double)
        )

    def forward(self, y):
        """
        Calculates distributions of latent variables.
        @param: y [n_timesteps, n_dim_obs] array-like
        @return: smoothed_state_means, smoothed_state_covariances,
        kalman_smoothing_gains
        """
        return self.smoother.smooth_torch(y)

    def predict_log_proba(self, batch):
        """
        batch is `torch.Tensor` of size `b x length x n_outs x 1` - that is
        4-dimensional.
        @return log_proba: torch.Tensor (b, )
        """
        return self.smoother.log_pred_density(batch)

    def training_step(self, batch, batch_idx):
        try:
            mu, p, h = self.smoother.smooth_torch(batch[1])
            m_step_result = self.smoother.maximization(batch[1], mu, p, h)
            self.smoother.update_parameters_from_tensors(*m_step_result)

            data_negloglikelihood = -self.smoother.loglikelihood_torch(batch[1])
            mean_frob_norm_sum = self.smoother.frob_diff_a().mean()
            loss = data_negloglikelihood + mean_frob_norm_sum
            self.log('aug_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('nll', data_negloglikelihood.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('mean_frob_norm_sum', mean_frob_norm_sum.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        except RuntimeError as err:
            print(f'Throwing err {err}')
            self.log('aug_loss', math.nan, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return self.dummy_param

    def configure_optimizers(self):
        # Fake optimizer, does nothing
        optimizer = torch.optim.Adam([self.dummy_param], lr=0.1)
        return optimizer

    def validation_step(self, batch, batch_idx):
        try:
            data_negloglikelihood = -self.smoother.loglikelihood_torch(batch[1])
            mean_frob_norm_sum = self.smoother.frob_diff_a().mean()
            val_loss = data_negloglikelihood + mean_frob_norm_sum
            self.log('val_nll', data_negloglikelihood.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_aug_loss', val_loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return val_loss
        except RuntimeError as err:
            print(f'Throwing err {err}')
            self.log('val_nll', math.nan, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_aug_loss', math.nan, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            

    def test_step(self, batch, batch_idx):
        data_negloglikelihood = -self.smoother.loglikelihood_torch(batch[1])
        self.log('val_nll', data_negloglikelihood.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return data_negloglikelihood

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs=100, verbose=True):
        """
        Fits a single class based on given data.
        @return best_model_path: path to the checkpoint with best performing model.
        """
        early_stop_callback = EarlyStopping(
            monitor='val_aug_loss',
            min_delta=0.000,
            patience=3,
            verbose=verbose,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_aug_loss',
            save_top_k=1,
            mode='min',
        )

        progress_bar_refresh_rate = 10
        if not verbose:
            progress_bar_refresh_rate = 0

        self.trainer = Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback,
                       early_stop_callback],
            progress_bar_refresh_rate=progress_bar_refresh_rate
        )
        self.trainer.fit(self, train_loader, val_loader)
        return self.trainer.checkpoint_callback.best_model_path

    def eval_elpd(self, batch):
        """
        Calculates log pointwise predictive density of the model given the batch 
        which serves as an estimate of expected log predictive density for
        the current model (set of parameters).
        """
        num_samples = batch[0].size(0)
        elpd_est = self.smoother.loglikelihood_torch(batch[1])
        return elpd_est, num_samples

    def eval_elpd_ldr(self, val_loader: DataLoader):
        """
        Computes elpd from the given val_loader for one instance of parameters.
        @return: elpd (scalar) of current model given val_loader data.
        """
        results = [self.eval_elpd(batch) for batch in val_loader]
        factored_elpds = [elpd * num_samples for elpd, num_samples in results]
        N = sum([num_samples for _, num_samples in results])
        elpd_weighted_avg = sum(factored_elpds)/N
        return elpd_weighted_avg


class LitEMPyKalman(LitEMBase):
    """
    ML System for HGMM model using PyKalman smoother.
    @param length: length of dataset.
    @param num_states: number of hidden states to use.
    @param num_outputs: number of outputs of the observable in the dataset.
    @param kappa: continuity preference strength - precision (inverse variance)
    of the parameter gaussian process.
    """
    def __init__(self, length, num_states, num_outputs, kappa):
        super().__init__()
        self.smoother = SmoothKalman(length=length,
                                     num_states=num_states,
                                     num_outputs=num_outputs,
                                     cont_precision=kappa)
        self.save_hyperparameters()


class LitEMTorch(LitEMBase):
    """
    ML System for HGMM model using torch modal state-space kalman.
    @param length: length of dataset.
    @param num_states: number of hidden states to use.
    @param num_outputs: number of outputs of the observable in the dataset.
    @param kappa: continuity preference strength - precision (inverse variance)
    of the parameter gaussian process.
    """
    def __init__(self, length, num_states, num_outputs, kappa):
        super().__init__()
        self.smoother = ModelHMMTorch(length=length,
                                      num_states=num_states,
                                      num_outputs=num_outputs,
                                      kappa=kappa)
        self.save_hyperparameters()


if __name__ == "__main__":
    from datetime import datetime
    import json
    from pathlib import Path

    torch.manual_seed(53)
    dataset_name = 'Coffee'
    mode = 'ucr'
    data_container = DatasetContainer(dataset_name, mode=mode, dtype=torch.double)
    sample_length = data_container.data_len
    N_states = 9
    N_outputs = data_container.output_dim
    kappa = 5e3  # (36e3)**2
    batch_size = 1000
    max_epochs = 1000

    my_model = LitEMTorch(length=sample_length,
                          num_states=N_states,
                          num_outputs=N_outputs,
                          kappa=kappa)
    tr_ldr = DataLoader(data_container.dataset_train, batch_size=batch_size)
    val_ldr = DataLoader(data_container.dataset_test, batch_size=batch_size)
    chkpt_paths = my_model.fit(tr_ldr, val_ldr, max_epochs=max_epochs, verbose=True)

    outdir = Path('out')
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    filename = f'{dataset_name}_{mode}_{timestamp}.json'
    save_file_path = outdir / Path(filename)
    with open(save_file_path, 'w') as f:
        json.dump(chkpt_paths, f)

    test_data_sample = data_container.dataset_test.tensors[1][0:1, :]
    inference = my_model.smoother(test_data_sample)
    plt.plot(inference[2].squeeze().detach(), label='Inference')
    plt.plot(test_data_sample.squeeze().detach(), label='Data')
    plt.legend()
    plt.show()
    print('Done')

else:
    print('Importing smooth_kalman.py')
