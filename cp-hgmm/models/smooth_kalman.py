import torch
import torch.nn.functional as F
import math

import numpy as np
from numpy.random import randn

from models.utils.kronecker import sylvester
from models.utils.cholesky import per_batch_cholesky
from models.utils.plotting import plot_confidence

from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader, random_split
# from tqdm import tqdm

import pykalman
from pykalman.standard import KalmanFilter, _smooth, _filter  # , _loglikelihoods
from pykalman.utils import preprocess_arguments, get_params, array1d, array2d, check_random_state

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import functools

import copy

# from ignite.engine import Engine, Events
# from ignite.handlers import EarlyStopping
# from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
# from ignite.metrics import Average

# from ignite.contrib.handlers.tensorboard_logger import *

# from ignite.contrib.engines.common import setup_tb_logging

from models.utils.dataset_torch_preparator import DatasetContainer


class SmoothKalman(KalmanFilter):
    def __init__(self,
                 length=1,
                 num_states=1,
                 num_outputs=1,
                 cont_precision=4,
                 kwd_params=dict()):

        self.kappa = cont_precision
        if 'kappa' in kwd_params:
            self.kappa = kwd_params['kappa']

        if self.kappa < 0:  # TI Kalman variant
            a_tensor = (1/math.sqrt(num_states))*torch.randn([num_states, num_states], dtype=torch.double).detach().numpy()
            q_tensor = torch.eye(num_states, dtype=torch.double).detach().numpy()
            transition_offsets = torch.zeros([num_states], dtype=torch.double).detach().numpy()
            observation_offsets = torch.zeros([num_outputs], dtype=torch.double).detach().numpy()
        else:  # TV Kalman
            a_tensor = (1/math.sqrt(num_states))*torch.randn([length-1, num_states, num_states], dtype=torch.double).detach().numpy()
            q_tensor = torch.eye(num_states, dtype=torch.double).repeat(length-1, 1, 1).detach().numpy()
            transition_offsets = torch.zeros([length-1, num_states], dtype=torch.double).detach().numpy()
            observation_offsets = torch.zeros([length, num_outputs], dtype=torch.double).detach().numpy()

        mat_b = (1/math.sqrt(num_outputs*num_states))*torch.randn([num_outputs, num_states], dtype=torch.double).detach().numpy()
        mat_r = torch.eye(num_outputs, dtype=torch.double).detach().numpy()
        mat_mu0 = torch.randn(torch.Size([num_states]), dtype=torch.double).detach().numpy()
        mat_p0 = torch.eye(num_states, dtype=torch.double).detach().numpy()



        default_kalman_params = {
            'transition_matrices': a_tensor,
            'observation_matrices': mat_b,
            'transition_covariance': q_tensor,
            'observation_covariance': mat_r,
            'initial_state_mean': mat_mu0,
            'initial_state_covariance': mat_p0,
            'transition_offsets': transition_offsets,
            'observation_offsets': observation_offsets,
            'em_vars': 'all'
        }

        subset_params = {
            x: kwd_params[x]
            for x in kwd_params if x in default_kalman_params
        }

        merged_kalman_params = {**default_kalman_params, **subset_params}

        super(SmoothKalman, self).__init__(**merged_kalman_params)

    @property
    def length(self):
        if len(self.transition_matrices.shape) <= 2:
            return 1
        a_len = self.transition_matrices.shape[0]
        return a_len+1

    def state_dict(self):
        # return vars(self)
        state = {
            'kappa': self.kappa,
            'transition_matrices': self.transition_matrices,
            'transition_covariance': self.transition_covariance,
            'observation_matrices': self.observation_matrices,
            'observation_covariance': self.observation_covariance,
            'initial_state_mean': self.initial_state_mean,
            'initial_state_covariance': self.initial_state_covariance,
            'transition_offsets': self.transition_offsets,
            'observation_offsets': self.observation_offsets
        }
        return state

    def load_state_dict(self, dictionary, **kwargs):
        self.kappa = dictionary['kappa']
        self.transition_offsets = dictionary['transition_offsets']
        self.observation_offsets = dictionary['observation_offsets']
        self.update_parameters_from_np(
            dictionary['transition_matrices'],
            dictionary['transition_covariance'],
            dictionary['observation_matrices'],
            dictionary['observation_covariance'],
            dictionary['initial_state_mean'],
            dictionary['initial_state_covariance']
        )

    def trim_length(self, y: torch.Tensor) -> torch.Tensor:
        '''For TI model, y is returned untouched. If y is longer than 
        model, shorten it to the model\'s length.
        '''
        assert type(y) is torch.Tensor, 'trim_length: Expected tensor input.'
        assert y.dim() == 4, 'trim_length: Expected 4-dimensional input.'

        if self.length == 1:  # Time-Invariant version doesn't trim
            return y

        input_len = y.size(1)
        if input_len > self.length:
            return y[:, 0:self.length]

        # if input_len < self.length:
        #     pad_right = self.length - input_len
        #     y_dimenzed = y[None,...].squeeze(-1)
        #     y_padded = F.pad(y_dimenzed, (0, 0, 0, pad_right), mode='replicate').squeeze(0).unsqueeze(-1)
        #     return y_padded

        return y

    def apply_torch(self, func, y):
        """Applies function on `torch.Tensor` batched (dim 0 is batch) y,
        y is supposed to be 4-dimensional (b*Lenghth*Outs*1). Expects the 
        func to return an iterable.
        Returns list of results (Tensored) corresponding to each batch.
        """
        assert type(y) is torch.Tensor, 'trim_length: Expected tensor input.'
        assert y.dim() == 4, 'smooth_torch: Expected 4-dimensional input.'

        y = self.trim_length(y)
        y = y.squeeze(-1)
        y_np = y.detach().numpy()
        result_list = []
        for y_i in y_np:
            result = func(y_i)
            result_cast = [torch.tensor(item, dtype=torch.double) for item in result]
            result_list.append(result_cast)

        return result_list

    def filter(self, y):
        Z = self._parse_observations(y)

        (predicted_state_means, predicted_state_covariances, kalman_gains,
         filtered_state_means, filtered_state_covariances) = (_filter(
             self.transition_matrices, self.observation_matrices, self.transition_covariance,
             self.observation_covariance, self.transition_offsets, self.observation_offsets,
             self.initial_state_mean, self.initial_state_covariance, Z))

        return (predicted_state_means, predicted_state_covariances, kalman_gains,
                filtered_state_means, filtered_state_covariances)

    def smooth_torch(self, y):
        """Calculates smooth estimates of batched (dim 0 is batch) y as
        `torch.Tensor`, y is supposed to be 4-dimensional `(b*Lenghth*Outs*1)`.
        """
        assert type(y) is torch.Tensor, 'trim_length: Expected tensor input.'
        assert y.dim() == 4, 'smooth_torch: Expected 4-dimensional input.'

        mu_list = []
        cov_list = []
        kal_gains_list = []
        y = self.trim_length(y)
        y = y.squeeze(-1)
        y_np = y.detach().numpy()

        for y_i in y_np:
            (smoothed_state_means, smoothed_state_covariances,
             kalman_smoothing_gains) = self.smooth(y_i)
            mu_list.append(torch.from_numpy(smoothed_state_means))
            cov_list.append(torch.from_numpy(smoothed_state_covariances))
            kal_gains_list.append(torch.from_numpy(kalman_smoothing_gains))

        mu_tensor = torch.stack(mu_list, dim=0).unsqueeze(-1)
        cov_tensor = torch.tensor(cov_list[0], dtype=torch.double)
        kal_gains_tensor = torch.tensor(kal_gains_list[0], dtype=torch.double)
        return mu_tensor, cov_tensor, kal_gains_tensor

    def smooth(self, X):
        """Apply the Kalman Smoother

        Apply the Kalman Smoother to estimate the hidden state at time
        :math:`t` for :math:`t = [0...n_{\\text{timesteps}}-1]` given all
        observations.  See :func:`_smooth` for more complex output

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]` is masked, then `X[t]` will be
            treated as a missing observation.

        Returns
        -------
        smoothed_state_means : [n_timesteps, n_dim_state]
            mean of hidden state distributions for times [0...n_timesteps-1]
            given all observations
        smoothed_state_covariances : [n_timesteps, n_dim_state]
            covariances of hidden state distributions for times
            [0...n_timesteps-1] given all observations
        kalman_smoothing_gains : [n_timesteps-1, n_dim_state, n_dim_state] array
            Kalman Smoothing correction matrices for times [0...n_timesteps-2]
        """
        Z = self._parse_observations(X)

        (predicted_state_means, predicted_state_covariances, _,
         filtered_state_means, filtered_state_covariances) = (_filter(
             self.transition_matrices, self.observation_matrices, self.transition_covariance,
             self.observation_covariance, self.transition_offsets, self.observation_offsets,
             self.initial_state_mean, self.initial_state_covariance, Z))
        (smoothed_state_means, smoothed_state_covariances,
         kalman_smoothing_gains) = (_smooth(self.transition_matrices,
                                            filtered_state_means,
                                            filtered_state_covariances,
                                            predicted_state_means,
                                            predicted_state_covariances))
        return (smoothed_state_means, smoothed_state_covariances,
                kalman_smoothing_gains)

    def predict_output_torch(self,
                            y,
                            predicted_state_means=None,
                            predicted_state_covariances=None):
        assert type(y) is torch.Tensor, 'trim_length: Expected tensor input.'
        assert y.dim() == 4, 'smooth_torch: Expected 4-dimensional input.'
        y = self.trim_length(y)

        if predicted_state_means is None or predicted_state_covariances is None:
            filter_list = self.apply_torch(self.filter, y)
            predicted_state_means_list = [item[0] for item in filter_list]
            predicted_state_covariances_list = [item[1] for item in filter_list]

        predicted_state_means_tensor = torch.stack(predicted_state_means_list, dim=0).unsqueeze(-1)
        predicted_state_covariances_tensor = predicted_state_covariances_list[0]
        observation_matrices_tensor = torch.tensor(self.observation_matrices, dtype=torch.double)
        observation_covariance_tensor = torch.tensor(self.observation_covariance, dtype=torch.double)

        y_hat = observation_matrices_tensor @ predicted_state_means_tensor
        sig = observation_matrices_tensor @ predicted_state_covariances_tensor @ observation_matrices_tensor.T + observation_covariance_tensor
        return y_hat, sig

    def log_pred_density(self,
                     y,
                     predicted_state_means=None,
                     predicted_state_covariances=None):
        '''Args are `torch.Tensor` of size `b x N x n_outs x 1` - that is 
        4-dimensional.'''
        assert type(y) is torch.Tensor, 'Expected tensor input.'
        assert y.dim() == 4, 'Expected 4-dimensional input.'
        y = self.trim_length(y)

        y_hat, sig = self.predict_output_torch(
            y, predicted_state_means, predicted_state_covariances)
        sig_chol = per_batch_cholesky(sig)
        nu = y - y_hat
        loader = DataLoader(nu)
        log_lik_list = []
        for nu_t in loader:
            nu_t_squeezed = nu_t.squeeze(0).squeeze(-1)
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(nu_t_squeezed.size()),
                scale_tril=sig_chol
            )
            log_lik = mvn.log_prob(nu_t_squeezed)
            log_lik_list.append(log_lik)
        return torch.stack(log_lik_list)

    def loglikelihood_torch(self,
                            y,
                            predicted_state_means=None,
                            predicted_state_covariances=None):
        """
        Calculates estimate of ELPD in the form of the log pointwise predictive density
        """
        assert type(y) is torch.Tensor, 'trim_length: Expected tensor input.'
        assert y.dim() == 4, 'smooth_torch: Expected 4-dimensional input.'
        results = self.log_pred_density(y, predicted_state_means, predicted_state_covariances)
        return results.mean()

    def _initialize_parameters(self, arguments=None):
        """Retrieve parameters if they exist, else replace with defaults"""
        n_dim_state, n_dim_obs = self.n_dim_state, self.n_dim_obs

        arguments = get_params(super(SmoothKalman, self))
        defaults = {
            'transition_matrices': np.eye(n_dim_state),
            'transition_offsets': np.zeros(n_dim_state),
            'transition_covariance': np.eye(n_dim_state),
            'observation_matrices': np.eye(n_dim_obs, n_dim_state),
            'observation_offsets': np.zeros(n_dim_obs),
            'observation_covariance': np.eye(n_dim_obs),
            'initial_state_mean': np.zeros(n_dim_state),
            'initial_state_covariance': np.eye(n_dim_state),
            'random_state': 0,
            'em_vars': [
                'transition_covariance',
                'observation_covariance',
                'initial_state_mean',
                'initial_state_covariance'
            ],
        }
        converters = self._get_param_converters()

        parameters = preprocess_arguments([arguments, defaults], converters)

        return (
            parameters['transition_matrices'],
            parameters['transition_offsets'],
            parameters['transition_covariance'],
            parameters['observation_matrices'],
            parameters['observation_offsets'],
            parameters['observation_covariance'],
            parameters['initial_state_mean'],
            parameters['initial_state_covariance']
        )

    def _get_param_converters(self):
        converters = {
            'transition_matrices': array2d,
            'transition_offsets': array1d,
            'transition_covariance': array2d,
            'observation_matrices': array2d,
            'observation_offsets': array1d,
            'observation_covariance': array2d,
            'initial_state_mean': array1d,
            'initial_state_covariance': array2d,
            'random_state': check_random_state,
            'n_dim_state': int,
            'n_dim_obs': int,
            'em_vars': lambda x: x,
        }
        return converters

    def maximization(self, y, x, p, h, a=None):
        """y is expected to be 4-dimensional (b*Length*Outs*1),
        x is 4-dims (b*Length*States*1),
        p is 3-dims (Length*States*States) symmetric positive definite
        (will try to handle pos. semi-definite),
        h is 3-dims (Length-1*States*States),
        a (optional) is 3-dims (Length-1*States*States).
        """
        xtt_outer = x @ x.permute(0, 1, 3, 2)
        c_xtxt = (p + xtt_outer).mean(dim=0)
        c_xtxt = (c_xtxt + c_xtxt.permute(0, 2, 1))/2
        c_xtxt_inv = c_xtxt.pinverse()
        # c_xtxt_inv = (c_xtxt_inv + c_xtxt_inv)/2
        c_xtxt_chol = per_batch_cholesky(c_xtxt).tril()
        xttm1_outer = x[:, 1:, :, :] @ (x[:, :-1, :, :].permute(0, 1, 3, 2))
        # size_ttm1 = xttm1_outer.size(1)
        ptcommatm1 = p[1:, :, :] @ h.permute(0, 2, 1)
        c_xtxtm1 = (ptcommatm1 + xttm1_outer).mean(dim=0)
        # a_tfrom1 = c_xtxtm1 @ c_xtm1xtm1^-1
        a_tfrom1 = a
        if a is None:
            # a_tfrom1 = (c_xtxtm1.permute(0, 2, 1).cholesky_solve(c_xtxt_chol[:-1])).permute(0, 2, 1) # t = 0 is skipped
            a_tfrom1 = c_xtxtm1 @ c_xtxt_inv[:-1]
            # q_tfrom1 = c_xtxt[1:] - (c_xtxtm1 @ c_xtxt_inv[:-1] @ (c_xtxtm1.permute(0, 2, 1)))
            # q_tfrom1 = (q_tfrom1 + q_tfrom1.permute(0, 2, 1))/2
            q_tfrom1 = c_xtxt[1:] - (c_xtxtm1 @ (c_xtxtm1.permute(0, 2, 1).cholesky_solve(c_xtxt_chol[:-1])))

            if self.kappa < 0:
                c_xtxtm1_mean = c_xtxtm1.mean(dim=0)
                c_xtxt_mean = c_xtxt.mean(dim=0)
                c_xtxt_mean_inv = c_xtxt_mean.pinverse()
                a_tfrom1 = c_xtxtm1_mean @ c_xtxt_mean_inv
                q_tfrom1 = c_xtxt_mean - (c_xtxtm1_mean @ (c_xtxtm1_mean.permute(1, 0).cholesky_solve(c_xtxt_mean.cholesky())))
                # a_tfrom1 = a_tfrom1.unsqueeze(0)
                # q_tfrom1 = q_tfrom1.unsqueeze(0)

        # cxtxtm1atT = c_xtxtm1 @ a_tfrom1.permute(0, 2, 1)
        # cxtxtm1atT_sym = (cxtxtm1atT + cxtxtm1atT.permute(0, 2, 1))
        # q_tfrom1 = c_xtxt[1:] - cxtxtm1atT_sym + (a_tfrom1 @ a_tfrom1.permute(0, 2, 1).cholesky_solve(c_xtxt_chol[:-1]))

        if self.kappa > 0:
            a_t_list = []
            a_t_cont = a_tfrom1[0]

            a_syl = self.kappa * q_tfrom1[0]
            b_syl = c_xtxt[0]
            q_syl = c_xtxtm1[0] + self.kappa * q_tfrom1[0] @ a_tfrom1[1]
            a_t_cont = sylvester(a_syl, b_syl, q_syl)
            a_t_list.append(a_t_cont)

            for i in range(1, q_tfrom1.size(0)-1):
                a_syl = 2 * self.kappa * q_tfrom1[i]
                b_syl = c_xtxt[i]
                q_syl = c_xtxtm1[i] + self.kappa * q_tfrom1[i] @ (a_t_cont + a_tfrom1[i+1])
                a_t_cont = sylvester(a_syl, b_syl, q_syl)
                a_t_list.append(a_t_cont)

            a_syl = self.kappa * q_tfrom1[-1]
            b_syl = c_xtxt[-2]
            q_syl = c_xtxtm1[-1] + self.kappa * q_tfrom1[-1] @ a_t_cont
            a_t_cont = sylvester(a_syl, b_syl, q_syl)
            a_t_list.append(a_t_cont)
            a_tfrom1 = torch.stack(a_t_list, 0)

        c_yy = (y @ y.permute(0, 1, 3, 2)).mean(dim=0)
        c_yy = (c_yy + c_yy.permute(0, 2, 1))/2
        c_yx = (y @ x.permute(0, 1, 3, 2)).mean(dim=0)
        # b_t = c_yx @ c_xtxt_inv
        b_t = c_yx.permute(0, 2, 1).cholesky_solve(c_xtxt_chol).permute(0, 2, 1)
        r_t = c_yy - c_yx @ c_xtxt_inv @ c_yx.permute(0, 2, 1)
        # try:
        #     r_t.cholesky()
        # except RuntimeError as err:
        #     r_t = c_yy - b_t @ (c_yx.permute(0, 2, 1))
        #     r_t.cholesky()

        b = b_t.mean(dim=0)
        r = r_t.mean(dim=0)
        r = (r + r.T)/2

        mu0 = x[:, 0, :, :].mean(dim=0)
        eps0 = x[:, 0:1, :, :] - mu0
        eps0_outer = eps0 @ eps0.permute(0, 1, 3, 2)
        p0 = (p[0, :, :] + eps0_outer).mean(dim=0)
        p0 = (p0 + p0.permute(0, 2, 1))/2

        return a_tfrom1, q_tfrom1, b, r, mu0, p0

    def update_parameters_from_tensors(self, a_t, q_t, b, r, mu0, p0):
        '''Args are `torch.tensors`, detached and converted to `np.ndarray` are
        then stored to the objects attributes.'''
        self.update_parameters_from_np(
            a_t.detach().numpy(),
            q_t.detach().numpy(),
            b.detach().numpy(),
            r.detach().numpy(),
            mu0.squeeze(1).detach().numpy(),
            p0.squeeze(0).detach().numpy()
        )

    def update_parameters_from_np(self, a_t, q_t, b, r, mu0, p0):
        '''Args are `np.ndarray`, then stored to the objects attributes.'''
        self.transition_matrices = a_t
        self.transition_covariance = q_t
        self.observation_matrices = b
        self.observation_covariance = r
        self.initial_state_mean = mu0
        self.initial_state_covariance = p0

    def train_em_step(self, engine, batch):
        '''training steps, call fit_em here, returned value will be stored in
        engine.state.output'''
        mu, p, h = self.smooth_torch(batch)
        try:
            m_step_result = self.maximization(batch, mu, p, h)
        except RuntimeError as err:
            print(f'Original error: {err}')
        self.update_parameters_from_tensors(*m_step_result)
        try:
            data_negloglikelihood = -self.loglikelihood_torch(batch)
        except RuntimeError as err:
            print(f'Original error: {err}')
        mean_frob_norm_sum = self.frob_diff_a().mean()
        loss = data_negloglikelihood  # + mean_frob_norm_sum  #+ self.frob_a().mean()
        return {'nll': loss.item(),
                'mean_frob_norm_sum': mean_frob_norm_sum.item()}

    def frob_diff_a(self):
        """ Returns sequence of frob. norms of differences of the sequence A_t.
        """
        a_t = torch.from_numpy(self.transition_matrices)
        if a_t.dim() <=2:
            return torch.tensor(0, dtype=torch.double)
        dif = a_t[:-1] - a_t[1:]
        return torch.norm(dif, p='fro', dim=(1, 2))

    def frob_a(self):
        """ Returns sequence of frob. norms of the sequence A_t.
        """
        a_t = torch.from_numpy(self.transition_matrices)
        return torch.norm(a_t, p='fro', dim=(1, 2))

    def r2_score(self, dataset):
        """Calculates mean R2 score of the model on the given dataset
        averaged over batches in dataset.
        """
        y = self.trim_length(dataset)
        y_squeezed = y.squeeze(-1).detach().numpy()

        y_hat, sigma = self.predict_output_torch(y)
        y_hat_squeezed = y_hat.squeeze(-1).detach().numpy()
        scores = []
        for y_sample, y_hat_sample in zip(y_squeezed, y_hat_squeezed):
            scores.append(r2_score(y_sample, y_hat_sample))
        scores_np = np.array(scores)
        return scores_np.mean(), scores_np.std(ddof=1)

    def plot_latent(self, y, y_noiseless=None, time=None):
        if time is None:
            time = np.arange(y.squeeze().size(0))
        y = self.trim_length(y)
        mu, p, h = self.smooth_torch(y)
        fig, axes = plt.subplots(nrows=2, figsize=(12, 9))

        y_hat, sig = self.predict_output_torch(y)

        y_sqz = y.squeeze(-1).squeeze(0).detach()
        y_hat_sqz = y_hat.squeeze(-1).squeeze(0).detach()
        mu_sqz = mu.squeeze(-1).squeeze(0).detach()

        labels = []
        if y_noiseless is not None:
            axes[0].plot(y_noiseless.squeeze().detach(), lw=1, linestyle='--')
            labels.extend([r'$true y_{0}$'.format(i) for i in range(y_sqz.size(-1))])
        plot_confidence(time, y_hat.squeeze(-1).squeeze(0).detach(), sig.detach(), ax=axes[0])
        labels.extend([r'$\hat y_{0} \pm 2\sigma$'.format(i) for i in range(y_sqz.size(-1))])
        axes[0].plot(time, y.squeeze().detach(), lw=1)
        labels.extend([r'$y_{0}$'.format(i) for i in range(y_sqz.size(-1))])
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Magnitude')
        axes[0].set_xlim(time[0], time[-1])
        axes[0].grid(True)
        axes[0].legend(labels)

        # axes[1].plot(time, mu.squeeze().detach())
        plot_confidence(time, mu_sqz, p.detach(), ax=axes[1])
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Magnitude')
        axes[1].set_xlim(time[0], time[-1])
        axes[1].grid(True)
        labels_x_hat = [r'$\hat x_{0} \pm 2\sigma$'.format(i) for i in range(mu_sqz.size(-1))]
        axes[1].legend(labels_x_hat)

        plt.tight_layout()

        plt.show()

    def print_info(self):
        print(
            f'Info TBD',
            f'\n------------------------------------------'
        )


class LitEMSystem(LightningModule):
    """
    ML System for HGMM model.
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
        self.dummy_param = torch.nn.Parameter(torch.rand([1, 1],
                                                         dtype=torch.double))
        self.save_hyperparameters()

    def state_dict(self):
        base_dict = super().state_dict()
        smoother_dict = self.smoother.state_dict()
        base_dict.update(smoother_dict)
        return base_dict

    def load_state_dict(self, state_dict, strict, **kwargs):
        super().load_state_dict(state_dict, strict=False)
        self.smoother.load_state_dict(state_dict)

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
        except RuntimeError:
            self.log('aug_loss', math.nan, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return self.dummy_param

    def configure_optimizers(self):
        # Fake optimizer, does nothing
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer

    def validation_step(self, batch, batch_idx):
        data_negloglikelihood = -self.smoother.loglikelihood_torch(batch[1])
        mean_frob_norm_sum = self.smoother.frob_diff_a().mean()
        val_loss = data_negloglikelihood + mean_frob_norm_sum
        self.log('val_nll', data_negloglikelihood.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_aug_loss', val_loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return data_negloglikelihood

    def test_step(self, batch, batch_idx):
        data_negloglikelihood = -self.smoother.loglikelihood_torch(batch[1])
        self.log('val_nll', data_negloglikelihood.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return data_negloglikelihood

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs=100):
        """
        Fits a single class based on given data.
        @return best_model_path: path to the checkpoint with best performing model.
        """
        early_stop_callback = EarlyStopping(
            monitor='val_nll',
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode='min'
        )

        early_nan_stop_callback = EarlyStopping(
            monitor='aug_loss',
            patience=5,
            verbose=True,
            mode='min',
            check_finite=True,
            check_on_train_epoch_end=True
        )

        early_stop_nll_callback = EarlyStopping(
            monitor='nll',
            patience=5,
            verbose=True,
            mode='min',
            check_on_train_epoch_end=True
        )

        self.trainer = Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback,
                       early_nan_stop_callback,
                       early_stop_nll_callback]
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

    def fit_all_classes(self, data_container: DatasetContainer, batch_size=1000, max_epochs=100):
        '''
        OBSOLETE!
        Runs training with early stopping and constructs default trainer for you.
        Trains over all classes in data_container, best model from last run are
        used as initial models for the next run, loops twice over models.
        @return: best models checkpoint paths
        '''
        # results = {label: checkpoint_path}
        results = {}
        data_loaders = {}
        for Xs, ys, label in data_container.class_train_datasets_generator():
            train_set, val_set = data_container._split_for_validation((Xs, ys), ratio=0.3)
            train_loader = DataLoader(train_set, batch_size=batch_size)
            val_loader = DataLoader(val_set, batch_size=batch_size)
            data_loaders[label] = (train_loader, val_loader)

        for label, (train_loader, val_loader) in data_loaders.items():
            self.fit(train_loader, val_loader, max_epochs=max_epochs)

        for label, (train_loader, val_loader) in data_loaders.items():
            results[label] = self.fit(train_loader, val_loader, max_epochs=max_epochs)

        return results


if __name__ == "__main__":
    from datetime import datetime
    import json
    from pathlib import Path

    torch.manual_seed(53)
    dataset_name = 'BasicMotions'
    mode = 'ucr'
    data_container = DatasetContainer(dataset_name, mode=mode, dtype=torch.double)
    sample_length = data_container.data_len
    N_states = 3
    N_outputs = data_container.output_dim
    cont_precision = 10.

    my_model = LitEMSystem(length=sample_length,
                           num_states=N_states,
                           num_outputs=N_outputs,
                           kappa=cont_precision)
    chkpt_paths = my_model.fit_all_classes(data_container, max_epochs=3)

    outdir = Path('out')
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    filename = f'{dataset_name}_{mode}_{timestamp}.json'
    save_file_path = outdir / Path(filename)
    with open(save_file_path, 'w') as f:
        json.dump(chkpt_paths, f)

    test_data_sample = data_container.dataset_test.tensors[1][0:1, :]
    # my_model.smoother.plot_latent(test_data_sample)

    for label in data_container.unique_labels:
        loaded_model = LitEMSystem.load_from_checkpoint(chkpt_paths[data_container.unique_labels[0]])
        loaded_model.smoother.plot_latent(test_data_sample)

    # y_hat, sigma = my_model.smoother.predict_output_torch(test_data_sample)

    # N_epochs = 10
    # batch_size = 80

    # sample_length = 100
    # N_samples = 10
    # N_states = 5
    # cont_precision = 10

    # # Dummy data to fit,
    # # a good example was with time 0:10, 380 sample_length, 2 N_states,
    # # 20 N_samples and 0.35 std of noise and generator [sin(7x), cos(5x)]
    # time_vector = torch.linspace(0, 3, sample_length, dtype=torch.double).unsqueeze(0)
    # x_tensor = time_vector.repeat(N_samples, 1)
    # # y_noiseless = torch.stack([
    # #     torch.sin(7 * x_tensor) + torch.cos(3 * x_tensor),
    # #     torch.cos(5 * x_tensor)
    # # ],
    # #                           dim=2).unsqueeze(-1)

    # y_noiseless = torch.stack([
    #     torch.sin(3 * x_tensor),
    #     torch.cos(5 * x_tensor)
    # ],
    #     dim=2).unsqueeze(-1)

    # # y_noiseless = torch.tensor([[1., 0.], [0.1, 0.5], [0.1, 0.1]], dtype=torch.double) @ y_noiseless
    # # y_noiseless = torch.tensor([[0.5, 0.5]], dtype=torch.double) @ y_noiseless
    # y = y_noiseless + (0.15 * torch.randn(y_noiseless.size()))
    # y_sample = y[0:1]
    # y_sample_np = y_sample.squeeze(0).squeeze(-1).detach().numpy()
    # N_outputs = y.size()[-2]



    # my_model = SmoothKalman(length=sample_length,
    #                         num_states=N_states,
    #                         num_outputs=N_outputs,
    #                         cont_precision=cont_precision)

    # res = my_model.fit_by_ignite_wrap(y, n_epochs=N_epochs, batch_size=batch_size, model_name='Unit_test')
    # # res = my_model.fit_em(y, n_epochs=N_epochs, batch_size=batch_size, progressbar=True)
    # # # res = my_model.fit_bp(y, n_epochs=N_epochs, batch_size=batch_size, continuity_pref=continuity_pref, lr=0.02, progressbar=True)
    # # data_negloglikmean_history, frob_norm_sum_history, models_history = res
    # # data_negloglikmean_history = res['mean_loglik_history']
    # # frob_norm_sum_history = res['mean_frob_history']
    # # models_history = res['model_parameters_history']

    # data_negloglikmean_history = res
    # my_model.plot_latent(y_sample)

    # (smoothed_state_means, smoothed_state_covariances,
    #     kalman_smoothing_gains) = my_model.smooth(y_sample_np)
    # y_hat, sigma = my_model.predict_output_torch(y_sample)

    # fig, axes = plt.subplots(nrows=4, figsize=(12, 9))

    # time = time_vector.squeeze().detach().numpy()
    # y_hat_squeezed = y_hat.squeeze(0).squeeze(-1)
    # sigma_squeezed = sigma.squeeze(0)
    # plot_kwargs = {'label': r'$\hat y \pm 2\sigma$'}
    # plot_confidence(time, y_hat_squeezed, sigma_squeezed, ax=axes[0], plot_kwargs=plot_kwargs)
    # # axes[0].plot(time, y_hat.squeeze().detach(), lw=2, label='Y_hat')
    # axes[0].plot(time, y_sample_np, lw=1, label=r'$y_{\epsilon}$')
    # axes[0].plot(time, y_noiseless[0].squeeze().detach(), lw=1, linestyle='--', label=r'$y$')
    # axes[0].set_xlabel('Sample')
    # axes[0].set_ylabel('Magnitude')
    # axes[0].grid(True)

    # covs_tensor = torch.from_numpy(smoothed_state_covariances)
    # plot_kwargs = {'label': r'$\hat x \pm 2\sigma$'}
    # # axes[1].plot(time, smoothed_state_means, plot_kwargs)
    # plot_confidence(time, smoothed_state_means, smoothed_state_covariances, ax=axes[1], plot_kwargs=plot_kwargs)
    # axes[1].set_xlabel('Sample')
    # axes[1].set_ylabel('Magnitude')
    # axes[1].grid(True)

    # frob_norm_seq = my_model.frob_diff_a()
    # axes[2].plot(frob_norm_seq.squeeze().detach(), label='Frob. A diff. seq.')
    # axes[2].set_xlabel('Sample')
    # axes[2].set_ylabel('Magnitude')
    # axes[2].grid(True)

    # # axes[3].plot(frob_norm_sum_history, label='Frob. norm sum history')
    # axes[3].plot(data_negloglikmean_history, label='Mean negloglik history')
    # axes[3].set_xlabel('Epoch')
    # axes[3].set_ylabel('Magnitude')
    # axes[3].grid(True)

    # [ax.legend() for ax in axes]
    # plt.tight_layout()
    # plt.show()

    # import datetime
    # from pathlib import Path
    # Path("out").mkdir(parents=True, exist_ok=True)

    # timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    # filename = f'out/{timestamp}_kappa{my_model.kappa}.pdf'
    # fig.savefig(filename, format='pdf')
else:
    print('Importing smooth_kalman.py')
