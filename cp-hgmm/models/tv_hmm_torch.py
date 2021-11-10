from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D

# from pytorch_lightning import seed_everything, LightningModule, Trainer
# from torch.utils.data import DataLoader, TensorDataset, Dataset
# from pytorch_lightning.callbacks import EarlyStopping

import math

from models.utils.kronecker import sylvester
from models.utils.cholesky import per_batch_cholesky
# from models.gp_smoother import SmootherSystem
# from models.utils.dataset_torch_preparator import split_for_validation

# import matplotlib.pyplot as plt
# from pathlib import Path


class ModelHMMTorch(nn.Module):
    '''Hidden Gauss-Markov model.'''

    def __init__(self,
                 length=1,
                 num_states=1,
                 num_outputs=1,
                 kappa=4,
                 dtype=None):
        super().__init__()
        if dtype is None:
            self.dtype = torch.double
        else:
            self.dtype = dtype

        self.register_buffer('num_states', torch.tensor(num_states), persistent=True)
        self.register_buffer('num_outputs', torch.tensor(num_outputs), persistent=True)
        self.register_buffer('length', torch.tensor(length), persistent=True)
        self.register_buffer('kappa', torch.tensor(kappa), persistent=True)
        # self.register_buffer('dtype', (dtype), persistent=True)
        w_a = 1/math.sqrt(num_states)
        w_b = 1/math.sqrt(num_outputs*num_states)
        if self.kappa < 0:  # TI Kalman variant
            self.a_tensor = nn.Parameter(
                w_a*torch.randn([num_states, num_states],
                                dtype=self.dtype)
                .expand(self.length, -1, -1)).clone()
            self.q_t_tensor = nn.Parameter(
                torch.eye(num_states, dtype=self.dtype)
                .expand(self.length, -1, -1)).clone()
        else:  # TV Kalman, Cont. TV including
            self.a_tensor = nn.Parameter(
                w_a*torch.randn(
                    [length-1, num_states, num_states],
                    dtype=self.dtype
                )
            )
            self.q_t_tensor = nn.Parameter(
                torch.eye(num_states,
                          dtype=self.dtype).repeat(length-1, 1, 1)
            )

        self.mat_b = nn.Parameter(
            w_b*torch.randn([num_outputs, num_states],
                            dtype=self.dtype)
        )
        self.mat_r = nn.Parameter(
            torch.eye(num_outputs, dtype=self.dtype)
        )
        self.mat_mu0 = nn.Parameter(
            torch.randn(torch.Size([num_states]), dtype=self.dtype)
        )
        self.mat_p0 = nn.Parameter(
            torch.eye(num_states, dtype=self.dtype)
        )

    def smooth_torch(self, y):
        """Calculates smooth estimates of batched (dim 0 is batch) y as
        `torch.Tensor`, y is supposed to be 4-dimensional `(b*Lenghth*Outs*1)`.
        """
        mu_tt_tensor, p_tt_tensor, yhat_from_fwd, sig_t_tensor = self(y)
        return mu_tt_tensor, p_tt_tensor, self.h_t

    def forward(self, y):
        """ Returns tuple of predictions of x, P, y, Sigma, H.
        @param y: (b, lentgh, dim, 1) observation in batch.
        """
        assert y.dim() == 4,\
            f'Data tensor `y` must be of dimension 4, given dim {y.dim()} of size {y.size()}.'
        mu_tt_tensor, yhat_from_fwd = self.roll_fwd(y)
        mu_t_tensor = self.roll_bwd(mu_tt_tensor)

        return mu_t_tensor,\
            self.p_t,\
            yhat_from_fwd,\
            self.sig_t

        # yhat_from_bwd = self.mat_b @ mu_t_tensor
        # r_act = self.mat_b @ self.p_t @ self.mat_b.T + self.mat_r
        # return mu_t_tensor,\
        #     self.p_t,\
        #     yhat_from_bwd,\
        #     r_act

    def roll_fwd(self, y):
        """ Returns tuple of predictions of mu, P, Sigma, H.
        @param y: (b, T, d, 1) observations.
        """
        batch_size = y.size(0)
        length = self.length

        b_t = self.mat_b

        y_t = y.permute(1, 0, 2, 3)

        self.precompute_covariances()

        y_hat_t = torch.zeros(length, batch_size, self.num_outputs, 1, dtype=self.dtype)
        mu_tt = torch.zeros(length, batch_size, self.num_states, 1, dtype=self.dtype)
        mu_ttm1 = torch.zeros(length, batch_size, self.num_states, 1, dtype=self.dtype)
        mu_ttm1[0] = self.mat_mu0.unsqueeze(1).repeat(batch_size, 1, 1)
        mu_tt[0] = mu_ttm1[0] + self.g_t[0] @ (y_t[0] - b_t @ mu_ttm1[0])
        y_hat_t[0] = b_t @ mu_ttm1[0]
        for t in range(1, length):  # (batch, out_dim, 1)
            # Time update (for the next iteration)
            mu_ttm1[t] = self.a_tensor[t-1] @ mu_tt[t-1]
            # Data update
            y_hat_t[t] = b_t @ mu_ttm1[t]
            mu_tt[t] = mu_ttm1[t] + self.g_t[t] @ (y_t[t] - y_hat_t[t])

        return mu_tt.permute(1, 0, 2, 3), y_hat_t.permute(1, 0, 2, 3)

    def roll_bwd(self, mu_tt_tensor):
        batch_size = mu_tt_tensor.size(0)
        n_timesteps = self.length
        mu_tt = mu_tt_tensor.permute(1, 0, 2, 3)

        mu_t = torch.zeros(n_timesteps, batch_size, self.num_states, 1, dtype=self.dtype)
        mu_t[-1] = mu_tt[-1]
        for t in reversed(range(n_timesteps - 1)):
            a_t = self.a_tensor[t]
            mu_t[t] = mu_tt[t] + self.h_t[t] @ (mu_t[t+1] - a_t @ mu_tt[t])
        return mu_t.permute(1, 0, 2, 3)

    def precompute_covariances(self) -> Tuple[Tensor, Tensor, Tensor]:
        '''Precomputes Kalman's gain G_t, `P_t|t` and `Sig_t`.
        All of these parameters are independent of data, they only depend
        on the model's parameters. Forward roll is used.
        Results are stored in class attributes and then also returned.'''

        n_timesteps = self.length
        q_t = self.q_t_tensor
        r_t = self.mat_r
        b_t = self.mat_b
        a_t = self.a_tensor
        # if self.a_tensor.dim() == 2:
        #     a_t = self.a_tensor.expand(n_timesteps, self.num_states, self.num_states)

        # if self.q_t_tensor.dim() == 2:
        #     q_t = self.q_t_tensor.expand(n_timesteps, self.num_states, self.num_states)

        # Forward recursion
        p_tt = torch.zeros(n_timesteps, self.num_states, self.num_states, dtype=self.dtype)
        p_ttm1 = torch.zeros(n_timesteps, self.num_states, self.num_states, dtype=self.dtype)
        sig_t = torch.zeros(n_timesteps, self.num_outputs, self.num_outputs, dtype=self.dtype)
        g_t = torch.zeros(n_timesteps, self.num_states, self.num_outputs, dtype=self.dtype)

        p_ttm1[0] = self.mat_p0
        g_t[0], p_tt[0], sig_t[0] = self.data_update_covariances(b_t, r_t, p_ttm1[0])
        for t in range(1, n_timesteps):
            p_ttm1[t] = a_t[t-1] @ p_tt[t-1] @ a_t[t-1].T + q_t[t-1]
            g_t[t], p_tt[t], sig_t[t] = self.data_update_covariances(b_t, r_t, p_ttm1[t])

        # Backward recursion

        p_t = torch.zeros(n_timesteps, self.num_states, self.num_states, dtype=self.dtype)
        h_t = torch.zeros(n_timesteps-1, self.num_states, self.num_states, dtype=self.dtype)
        p_t[-1] = p_tt[-1]
        predicted_state_covariances_inv = torch.linalg.pinv(p_ttm1, hermitian=True)
        for t in reversed(range(n_timesteps - 1)):
            predicted_state_covariance = p_ttm1[t + 1]
            predicted_state_covariance_inv = predicted_state_covariances_inv[t + 1]

            h_t[t] = p_tt[t] @ a_t[t].T\
                @ predicted_state_covariance_inv

            p_t[t] = p_tt[t] + h_t[t]\
                @ (p_t[t + 1] - predicted_state_covariance) @ h_t[t].T

        v_ttm1 = torch.zeros(n_timesteps-1, self.num_states, self.num_states, dtype=self.dtype)
        v_ttm1[-1] = (torch.eye(self.num_states)-g_t[-1] @ b_t) @ a_t[-2] @ p_tt[-2]
        factor = p_tt[1:] @ h_t
        for t in reversed(range(n_timesteps - 2)):
            v_ttm1[t] = factor[t] + h_t[t+1] @ (v_ttm1[t+1] - a_t[t] @ p_tt[t+1]) @ h_t[t].T
        # ---------------------------
        self.p_tt = p_tt
        self.p_ttm1 = p_ttm1
        self.h_t = h_t
        self.g_t = g_t
        self.sig_t = sig_t
        self.v_ttm1 = v_ttm1
        self.p_t = p_t

    def data_update_covariances(self, b_t, r_t, p_ttm1):
        '''Computes G_t (i.e. Kalman's gain), P_t|t and Sigma_t.'''
        sig_t = b_t @ p_ttm1 @ b_t.T + r_t
        g_t = p_ttm1 @ b_t.T @ torch.linalg.pinv(sig_t, hermitian=True)
        p_tt = p_ttm1 - g_t @ b_t @ p_ttm1
        return g_t, p_tt, sig_t

    def log_pred_density(self, y):
        """Calculates log predictive density.
        For pointwise estimate of the Expected log-predictive density, call mean.
        This calls forward inside.
        Args are `torch.Tensor` of size `b x N x n_outs x 1` - that is 
        4-dimensional.
        @returns: (b, N) of log predictive densities at given query points y.
        """
        assert y.dim() == 4, 'Expected 4-dimensional input.'
        x_hat, p, yhat_from_fwd, sig_tensor = self(y)
        # sig_chol_list = []
        sig_chol_tensor = per_batch_cholesky(sig_tensor)
        # for sig in sig_tensor:
        #     sig_chol_list.append(torch.cholesky(sig))
        # sig_chol_tensor = torch.stack(sig_chol_list)

        mvn = D.multivariate_normal.MultivariateNormal(
            yhat_from_fwd.squeeze(-1), scale_tril=sig_chol_tensor
        )
        lpd = mvn.log_prob(y.squeeze(-1))
        return lpd

    def loglikelihood_torch(self, y):
        """ELPD pointwise estimate. It's the mean of log-predictive density.
        """
        lpds = self.log_pred_density(y)
        return lpds.mean()

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
            q_tfrom1 = c_xtxt[1:] - (c_xtxtm1 @ (c_xtxtm1.permute(0,
                                     2, 1).cholesky_solve(c_xtxt_chol[:-1])))

            if self.kappa < 0:
                c_xtxtm1_mean = c_xtxtm1.mean(dim=0)
                c_xtxt_mean = c_xtxt.mean(dim=0)
                c_xtxt_mean_inv = c_xtxt_mean.pinverse()
                a_tfrom1 = c_xtxtm1_mean @ c_xtxt_mean_inv
                q_tfrom1 = c_xtxt_mean - \
                    (c_xtxtm1_mean @ (c_xtxtm1_mean.permute(1,
                     0).cholesky_solve(c_xtxt_mean.cholesky())))
                a_tfrom1 = a_tfrom1.expand(self.length, -1, -1)
                q_tfrom1 = q_tfrom1.expand(self.length, -1, -1)

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
                q_syl = c_xtxtm1[i] + self.kappa * \
                    q_tfrom1[i] @ (a_t_cont + a_tfrom1[i+1])
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
        b_t = c_yx.permute(0, 2, 1).cholesky_solve(
            c_xtxt_chol).permute(0, 2, 1)
        r_t = c_yy - c_yx @ c_xtxt_inv @ c_yx.permute(0, 2, 1)
        # try:
        #     r_t.cholesky()
        # except RuntimeError as err:
        #     r_t = c_yy - b_t @ (c_yx.permute(0, 2, 1))
        #     r_t.cholesky()

        b = b_t.mean(dim=0)
        r = r_t.mean(dim=0)
        r = (r + r.T)/2

        mu0 = x[:, 0, :, 0].mean(dim=0)
        eps0 = (x[:, 0, :, 0] - mu0).unsqueeze(-1)
        eps0_outer = eps0 @ eps0.permute(0, 2, 1)
        p0 = p[0, :, :] + eps0_outer.mean(dim=0)
        p0 = (p0 + p0.T)/2

        # mu0 = x[:, 0, :, :].mean(dim=0)
        # eps0 = x[:, 0:1, :, :] - mu0
        # eps0_outer = eps0 @ eps0.permute(0, 1, 3, 2)
        # p0 = (p[0, :, :] + eps0_outer).mean(dim=0)
        # p0 = (p0 + p0.permute(0, 2, 1))/2

        return a_tfrom1, q_tfrom1, b, r, mu0, p0

    def frob_diff_a(self):
        """ Returns sequence of frob. norms of differences of the sequence A_t.
        """
        a_t = self.a_tensor
        if a_t.dim() <= 2:
            return torch.tensor(0, dtype=torch.double)
        dif = a_t[:-1, :] - a_t[1:, :]
        factor = self.kappa / dif.size(0)
        return factor * torch.norm(dif, p='fro', dim=(1, 2))

    def update_parameters_from_tensors(self, a_t, q_t, b, r, mu0, p0):
        '''Args are `torch.tensors`, are
        then stored to the objects attributes.'''
        self.a_tensor.data = a_t
        self.q_t_tensor.data = q_t
        self.mat_b.data = b
        self.mat_r.data = r
        self.mat_mu0.data = mu0
        self.mat_p0.data = p0


if __name__ == "__main__":
    import numpy as np
    from models.utils.dataset_torch_preparator import DatasetContainer
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    torch.manual_seed(3)
    N_epochs = 1400
    batch_size = 1000
    N_states = 3
    kappa = -100

    dataset_name = 'Coffee'

    # now_str = datetime.now().strftime("%Y%m%dT%H%M%S")
    # model_root = Path(f'out/state_dicts/{dataset_name}_{now_str}')
    # model_root.mkdir(exist_ok=True)
    # log_root = Path(f'logs/tb_logs/{dataset_name}_{now_str}')
    # log_root.mkdir(exist_ok=True)

    dataset_container = DatasetContainer(dataset_name, mode='ucr')
    class_occurences = dataset_container.select_class_train(
        dataset_container.label_encoder.classes_[0]
    )[1]
    sample_length = dataset_container.data_len
    N_outputs = dataset_container.output_dim

    # dataset_container.plot_dataset()

    my_model = ModelHMMTorch(
        sample_length, N_states,
        N_outputs, kappa=kappa
    )
    # my_model(class_occurences)
    lpd = my_model.log_pred_density(class_occurences)
    print(f'ELPD pointwise: {lpd.mean()}')

    # run_fit_lightning(class_occurences, my_model)

    # y_sample = dataset_container.select_class(
    #     dataset_container.dataset_test,
    #     dataset_container.label_encoder.classes_[0]
    # )[0:1, :]

    # pred = my_model(y_sample)
    # mu, p, y_hat, sig = pred
    # data_negloglikelihood = -my_model.data_meanlikelihood(y_sample, y_hat, sig)
    # print(f'Test: neg. log-likelihood {data_negloglikelihood}')

    # fig, axes = plt.subplots(nrows=3, figsize=(12, 9))

    # y_mean = y_hat.squeeze(0).squeeze(-1)
    # time = np.arange(y_mean.size(0))
    # for out_id in np.arange(sig.size(-1)):
    #     mean = y_mean[:, out_id].detach().numpy()
    #     std = sig[:, out_id, out_id].detach().numpy()
    #     axes[0].fill_between(time, mean-std, mean+std,
    #                          alpha=0.5, label=f'{out_id} $\pm \sigma$')

    # axes[0].plot(y_hat.squeeze().detach(), lw=2, label='Y_hat')
    # axes[0].plot(y_sample.squeeze().detach(), lw=1, label='Y_data')
    # axes[0].set_xlabel('Sample')
    # axes[0].set_ylabel('Magnitude')
    # axes[0].grid(True)

    # axes[1].plot(mu.squeeze().detach(), label='X_hat')
    # axes[1].set_xlabel('Sample')
    # axes[1].set_ylabel('Magnitude')
    # axes[1].grid(True)

    # frob_norm_seq = my_model.frob_diff_a()
    # axes[2].plot(frob_norm_seq.squeeze().detach(), label='Frob. A diff. seq.')
    # axes[2].set_xlabel('Sample')
    # axes[2].set_ylabel('Magnitude')
    # axes[2].grid(True)

    # [ax.legend() for ax in axes]
    # plt.tight_layout()
    # plt.show()

    # # ax.savefig(filenames[0] + '.pdf', format='pdf')

    # plt.plot(my_model.a_tensor.squeeze().detach())
    # plt.show()
    print('Done')
else:
    print('Importing data_serializer.py serializer')
