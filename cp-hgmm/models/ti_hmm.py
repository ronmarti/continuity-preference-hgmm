import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
# from data_serializer import DatasetSerializerCSV

from utils.cholesky import per_batch_cholesky


class Model_hmm(nn.Module):
    def __init__(self, length, num_states, num_outputs):
        super(Model_hmm, self).__init__()
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.length = length

        # self.a_tensor = nn.Parameter((1/math.sqrt(num_states))*torch.randn([num_states, num_states]))
        self.a_tensor = nn.Parameter((1/num_states)*torch.randn([num_states, num_states]))
        self.register_parameter(f'a_tensor', self.a_tensor)

        # self.q_chtril_tensor = nn.Parameter(torch.tril(torch.eye(num_states)))
        q_rnd_sqrt = (1/num_states)*torch.randn([num_states, num_states])
        q_rnd_chol = per_batch_cholesky(q_rnd_sqrt @ q_rnd_sqrt.T)
        self.q_chtril_tensor = nn.Parameter(torch.tril(q_rnd_chol))
        self.register_parameter(f'q_chtril_tensor', self.q_chtril_tensor)

        # self.mat_p0_chtril = nn.Parameter(torch.tril(torch.eye(num_states)))
        p0_rnd_sqrt = torch.randn(num_states, num_states)
        p0_rnd_chol = (p0_rnd_sqrt @ p0_rnd_sqrt.T).cholesky()
        self.mat_p0_chtril = nn.Parameter(torch.tril(p0_rnd_chol))
        self.register_parameter('mat_p0_chtril', self.mat_p0_chtril)
        self.mat_mu0 = nn.Parameter(1+torch.randn(torch.Size([num_states])))
        self.register_parameter('mat_mu0', self.mat_mu0)
        self.mat_b = nn.Parameter((1/math.sqrt(num_outputs))*torch.randn([num_outputs, num_states]))
        self.register_parameter('mat_b', self.mat_b)
        # self.mat_r_chtril = nn.Parameter(torch.tril(1*torch.eye(num_outputs)))
        r_rnd_sqrt = (1/math.sqrt(num_outputs*num_states))*torch.randn(num_outputs, num_outputs)
        r_rnd_chol = per_batch_cholesky(r_rnd_sqrt @ r_rnd_sqrt.T)
        self.mat_r_chtril = nn.Parameter(torch.tril(r_rnd_chol))
        self.register_parameter('mat_r_chtril', self.mat_r_chtril)

        self.eye_states_size = nn.Parameter(torch.eye(self.num_states))
        self.register_buffer('eye_states_size_buffer', self.eye_states_size)

    def forward(self, y):
        """ Returns tuple of predictions of x, P, y, Sigma, H.
        """
        pred_fwd = self.roll_fwd(y)
        mu_tt_tensor, p_tt_tensor, yhat_from_fwd, sig_tensor, h_tensor = pred_fwd
        pred_bwd = self.roll_bwd(y)
        mutT, ptT = self.merge_estimates(mu_tt_tensor, p_tt_tensor, *pred_bwd)
        # yhat_from_fwd = self.mat_b @ mu_tt_tensor
        return mutT,\
            ptT,\
            yhat_from_fwd,\
            sig_tensor,\
            h_tensor

    def roll_fwd(self, y):
        """ Returns tuple of predictions of mu, P, Sigma, H.
        """
        mu_ttm1_list = []
        mu_tt_list = []
        p_tt_list = []
        sig_list = []
        h_t_list = []
        batch_size = y.size()[0]
        q_chtril_tensor = torch.tril(self.q_chtril_tensor)
        q_t = q_chtril_tensor @ q_chtril_tensor.T
        a_t = self.a_tensor
        y_permuted = y.permute(1, 0, 2, 3)

        # dataset = TensorDataset()
        loader = DataLoader(y_permuted[1:, :, :, :])
        mat_r = torch.tril(self.mat_r_chtril) @ torch.tril(self.mat_r_chtril).T
        p_0 = self.mat_p0_chtril @ self.mat_p0_chtril.T
        mu_0 = self.mat_mu0.unsqueeze(1).repeat(batch_size, 1, 1)
        y_0 = y_permuted[0, :, :, :]

        mu_tt, p_tt, sig = self.data_update(y_0, self.mat_b, mat_r, mu_0, p_0)
        sig_list.append(sig)
        mu_ttm1_list.append(mu_0)
        mu_tt_list.append(mu_tt)
        p_tt_list.append(p_tt)

        for y_t in loader:
            # Time update
            y_t = y_t.squeeze(0)
            # a_txp_tm1tm1 = a_t @ p_tt
            # p_ttm1 = torch.addmm(q_t, a_txp_tm1tm1, a_t.T)
            # h_t = a_txp_tm1tm1.cholesky_solve(per_batch_cholesky(p_ttm1)).T
            # h_t_list.append(h_t)

            p_ttm1 = q_t + a_t @ p_tt @ a_t.T
            p_ttm1 = (p_ttm1 + p_ttm1)/2
            h_t = p_tt @ a_t.T @ p_ttm1.pinverse()
            h_t_list.append(h_t)

            mu_ttm1 = a_t @ mu_tt
            mu_ttm1_list.append(mu_ttm1)
            # Data update
            mu_tt, p_tt, sig = self.data_update(y_t, self.mat_b, mat_r, mu_ttm1, p_ttm1)
            sig_list.append(sig)
            mu_tt_list.append(mu_tt)
            p_tt_list.append(p_tt)

        mu_ttm1_tensor = torch.stack(mu_ttm1_list).permute(1, 0, 2, 3)
        yhat_tensor = self.mat_b @ mu_ttm1_tensor
        mu_tt_tensor = torch.stack(mu_tt_list).permute(1, 0, 2, 3)
        p_tt_tensor = torch.stack(p_tt_list)

        return mu_tt_tensor,\
            p_tt_tensor,\
            yhat_tensor,\
            torch.stack(sig_list),\
            torch.stack(h_t_list)

    def data_update(self, y_t, b_t, r_t, mu_ttm1, p_ttm1):
        sig = r_t + (b_t @ p_ttm1 @ b_t.T)
        sig = (sig + sig.T)/2
        p_ttm1xbT = p_ttm1 @ b_t.T
        g = p_ttm1xbT.T.cholesky_solve(per_batch_cholesky(sig)).T
        imgb = torch.eye(self.num_states) - (g @ b_t)
        p_tt = imgb @ p_ttm1
        p_tt = (p_tt + p_tt.T)/2
        mu_tt = (imgb @ mu_ttm1) + (g @ y_t)
        return mu_tt, p_tt, sig

    def roll_bwd(self, y):
        """ Returns tuple of predictions of ksi, gamma.
        """
        ksi_tm1t_list = []
        gamma_tm1t_list = []
        batch_size = y.size()[0]
        q_chtril_tensor = torch.tril(self.q_chtril_tensor)
        q_t = torch.matmul(q_chtril_tensor, q_chtril_tensor.T)

        qinv_t = torch.pinverse(q_t)

        y_permuted = y.permute(1, 0, 2, 3)
        y_bwd = torch.flip(y_permuted[1:], [0])
        a_bwd = self.a_tensor
        qinv_bwd = qinv_t

        loader = DataLoader(y_bwd)
        mat_r = torch.matmul(torch.tril(self.mat_r_chtril),
                             torch.tril(self.mat_r_chtril).T)
        mat_rinv = mat_r.pinverse()
        gamma_tm1t = torch.zeros(self.num_states, self.num_states)
        gamma_tm1t_list.append(gamma_tm1t)
        ksi_tm1t = torch.zeros(self.num_states).unsqueeze(1).repeat(batch_size, 1, 1)
        ksi_tm1t_list.append(ksi_tm1t)
        for y_t in loader:
            # Data update
            y_t = y_t.squeeze(0)
            qinv_t = qinv_t
            bTxrinv = self.mat_b.T @ mat_rinv
            gamma_tt = gamma_tm1t + bTxrinv @ self.mat_b
            ksi_tt = ksi_tm1t + bTxrinv @ y_t

            # Time update
            pi_t = gamma_tt + qinv_t
            piinvxqinv = pi_t.pinverse() @ qinv_t

            gamma_tm1t = a_bwd.T @ qinv_t @ (self.eye_states_size - piinvxqinv) @ a_bwd
            ksi_tm1t = a_bwd.T @ piinvxqinv.T @ ksi_tt

            ksi_tm1t_list.append(ksi_tm1t)
            gamma_tm1t_list.append(gamma_tm1t)

        ksi_tm1t_tensor = torch.flip(torch.stack(ksi_tm1t_list).permute(1, 0, 2, 3), [1])

        return ksi_tm1t_tensor,\
            torch.flip(torch.stack(gamma_tm1t_list), [0])

    def merge_estimates(self, mu, p, ksi, gamma):
        p_inv = p.pinverse()
        ptT = (gamma + p_inv).pinverse()
        mutT = ptT @ (ksi + p_inv @ mu)
        return mutT, ptT

    def chol_update(self, a, b, c):
        """Calculates schur complement of a of matrix [[a, b.T], [b, c]].
        Returns cholesky lower triangular decomposition.
        """
        size = c.size(1)
        # trans = [i for i in range(c.dim()-2)]
        # trans.append(-1)
        # trans.append(-2)
        row1 = torch.cat([a, b.T], dim=1)
        row2 = torch.cat([b, c], dim=1)
        mat = torch.cat([row1, row2], dim=0)
        labc = per_batch_cholesky(mat)
        lc = labc[-size:, -size:]
        return lc

    def maximization(self, y, x, p, h):
        x_outer = x @ x.permute(0, 1, 3, 2)
        x_outer_m = x_outer.mean(dim=0).mean(dim=0)

        c_xtxt = p.mean(dim=0) + x_outer_m
        c_xtxt = (c_xtxt + c_xtxt.T)/2

        # p_inv = torch.cholesky_inverse(per_batch_cholesky(p))

        # p_inv = p.mean(dim=0).pinverse()
        # p_inv = (p_inv + p_inv.permuteT)/2
        # factor = 1/(1 + x_m.permute(0, 2, 1) @ p_inv @ x_m)
        # c_xtxt_inv = p_inv - factor * (p_inv @ xtt_outer @ p_inv)
        # c_xtxt_inv = ((c_xtxt_inv + c_xtxt_inv.permute(0, 2, 1))/2).mean(dim=0)

        c_xtxt_inv = c_xtxt.pinverse()
        c_xtxt_inv = (c_xtxt_inv + c_xtxt_inv.T)/2
        # c_xtxt_dif = c_by_pinv - c_xtxt_inv
        # c_xtxt = p + xtt_outer
        xttm1_outer = x[:, 1:, :, :] @ (x[:, :-1, :, :].permute(0, 1, 3, 2))
        xttm1_outer_m = xttm1_outer.mean(dim=0).mean(dim=0)
        ptcommatm1 = p[1:, :, :] @ h.permute(0, 2, 1)
        c_xtxtm1 = (ptcommatm1.mean(dim=0) + xttm1_outer_m)
        # c_xtxtm1 = ptcommatm1 + xttm1_outer
        # a_tfrom1 = c_xtxtm1 @ c_xtm1xtm1^-1
        # a_tfrom1 = (c_xtxtm1.T.cholesky_solve(per_batch_cholesky(c_xtxt))).T  # t = 0 is skipped
        a_tfrom1 = c_xtxtm1 @ c_xtxt_inv

        # a_test = c_xtxtm1.mean(dim=0) @ c_xtxt.mean(dim=0).pinverse()
        # q_test = c_xtxt.mean(dim=0) - (a_test @ (c_xtxtm1.mean(dim=0).T))

        # a_tfrom1 = c_xtxtm1 @ c_xtxt.pinverse()
        # q_tfrom1 = per_batch_cholesky(c_xtxt - (a_tfrom1 @ (c_xtxtm1.permute(1, 0))))
        q_sqrt = self.chol_update(c_xtxt, c_xtxtm1, c_xtxt)
        q_tfrom1 = q_sqrt @ q_sqrt.T
        q_compare = c_xtxt - c_xtxtm1 @ c_xtxt_inv @ c_xtxtm1.T
        q_t_dif = q_tfrom1 - q_compare
        # a_tfrom1 = a_tfrom1.mean(dim=0)  # Homogeneous case
        # q_tfrom1 = q_tfrom1.mean(dim=0)  # Homogeneous case

        # a_test_dif = a_tfrom1 - a_test
        # q_test_dif = q_tfrom1 - q_test
        # q_tfrom1 = c_xtxt[1:] - (c_xtxtm1 @ (c_xtxtm1.permute(0, 2, 1).cholesky_solve(c_xtxt[:-1].cholesky())))
        try:
            per_batch_cholesky(q_tfrom1)
        except RuntimeError as err:
            q_tfrom1 = per_batch_cholesky(q_tfrom1) @ per_batch_cholesky(q_tfrom1).permute(0, 2, 1)
            q_tfrom1.cholesky()

        c_yy = (y @ y.permute(0, 1, 3, 2)).mean(dim=0).mean(dim=0)
        # c_yy = (c_yy + c_yy.permute(0, 1, 3, 2))/2
        c_yx = (y @ x.permute(0, 1, 3, 2)).mean(dim=0).mean(dim=0)
        # b = (c_yx @ c_xtxt.pinverse())
        b = c_yx.T.cholesky_solve(per_batch_cholesky(c_xtxt)).T
        # r_old = (c_yy - b @ c_yx.permute(1, 0))
        r_sqrt = self.chol_update(c_xtxt, c_yx, c_yy)
        r = r_sqrt @ r_sqrt.T
        # r_dif = r_old - r
        try:
            per_batch_cholesky(r)
        except RuntimeError as err:
            r = per_batch_cholesky(r) @ per_batch_cholesky(r).permute(1, 0)
            r.cholesky()

        mu0 = x[:, 0, :, :].mean(dim=0)
        eps0 = x[:, 0, :, :] - mu0
        eps0_outer = eps0 @ eps0.permute(0, 2, 1)
        p0_sqrt = per_batch_cholesky((p[0, :, :] + eps0_outer).mean(dim=0))

        # return a_tfrom1, q_tfrom1, b, r, mu0, p0
        return a_tfrom1, q_sqrt, b, r_sqrt, mu0, p0_sqrt

    def update_parameters(self, a_t, q_t_chol, b, r_chol, mu0, p0_chol):
        self.a_tensor.data = a_t
        # try:
        #     q_t.cholesky()
        # except RuntimeError as err:
        #     print(f'Error Q: {err}')
        self.q_chtril_tensor.data = q_t_chol.tril()
        self.mat_b.data = b
        self.mat_r_chtril.data = r_chol.tril()
        # try:
        #     r.cholesky()
        # except RuntimeError as err:
        #     print(f'Error R: {err}')
        self.mat_mu0.data = mu0.squeeze(1)
        self.mat_p0_chtril.data = p0_chol.tril()
        if torch.any(self.mat_p0_chtril.data.diag() <= 1e-6):
            print('Ill posed P_0')
        # try:
        #     p0.squeeze(0).cholesky()
        # except RuntimeError as err:
        #     print(f'Error mat_p0_chtril: {err}')

    def data_meanlikelihood(self, y_batch, y_hat, sig):
        """ Log-likelihoood mean of the data given the predictive
        parameters of hidden state densities.
        """
        batch_size = y_batch.size()[0]
        # x, p, yhat, sig, h = pred
        nu = y_batch - y_hat
        log_lik_list = []
        loader = DataLoader(nu)
        for nu_t in loader:
            nu_t_squeezed = nu_t.squeeze(0).squeeze(-1)
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(nu_t_squeezed.size()),
                sig
            )
            log_lik = mvn.log_prob(nu_t_squeezed)
            log_lik_list.append(log_lik)
        return torch.mean(torch.stack(log_lik_list))

    def frob_a(self):
        """ Returns sequence of frob. norms of the sequence A_t.
        """
        return torch.norm(self.a_tensor, p='fro', dim=(0, 1))

    def fit_em(self, y, n_epochs=10, batch_size=100, progressbar=False):
        """ Fits model to given data y using EM algorithm.
        """
        # n_samples = y.size(0)
        # x_init = torch.rand(n_samples, self.length, self.num_states, 1)
        # p_init = 0.25*torch.eye(self.num_states).repeat(self.length, 1, 1)
        # h_init = 0.25*torch.eye(self.num_states).repeat(self.length-1, 1, 1)
        # max_result = self.maximization(y, x_init, p_init, h_init)
        # self.update_parameters(*max_result)

        tqdm_epochs = tqdm(range(n_epochs), disable=not progressbar)

        # dataset = TensorDataset(y) #only useful when sampling tuple of tensors
        loader = DataLoader(y, batch_size=batch_size)

        frob_norm_history = []
        data_negloglikmean_history = []
        early_stop = False
        for epoch in tqdm_epochs:
            if early_stop:
                break
            for y_minibatch in loader:
                pred = self(y_minibatch)
                mu, p, y_hat, sig, h = pred
                try:
                    m_step_result = self.maximization(y_minibatch, mu, p, h)
                except RuntimeError as err:
                    early_stop = True
                    print(f'Original error: {err}')
                    break
                self.update_parameters(*m_step_result)
                data_negloglikelihood = -self.data_meanlikelihood(y_minibatch, y_hat, sig)
                frob_norm = self.frob_a()
                loss = data_negloglikelihood  # + mean_frob_norm_sum  #+ self.frob_a().mean()
                tqdm_epochs.set_postfix(loss=loss.item())
            frob_norm_history.append(frob_norm)
            data_negloglikmean_history.append(data_negloglikelihood)

        return data_negloglikmean_history, frob_norm_history

    def fit_bp(self, y, n_epochs=10, batch_size=100, lr=0.01, progressbar=False):
        """ Fits model to given data y using BackPropagation algorithm.
        """
        tqdm_epochs = tqdm(range(n_epochs), disable=not progressbar)

        # dataset = TensorDataset(y) #only useful when sampling tuple of tensors
        loader = DataLoader(y, batch_size=batch_size)

        # loss_criterion = nn.MSELoss()
        opt = optim.Adam(self.parameters(), lr=lr)
        frob_norm_history = []
        data_negloglikmean_history = []
        early_stop = False
        for epoch in tqdm_epochs:
            if early_stop:
                break
            for y_minibatch in loader:
                pred = self(y_minibatch)
                mu, p, y_hat, sig, h = pred
                data_negloglikelihood = -self.data_meanlikelihood(y_minibatch, y_hat, sig)
                frob_norm = self.frob_a()
                loss = data_negloglikelihood
                opt.zero_grad()
                loss.backward()
                opt.step()
                tqdm_epochs.set_postfix(loss=loss.item())
            frob_norm_history.append(frob_norm)
            data_negloglikmean_history.append(data_negloglikelihood)

        return data_negloglikmean_history, frob_norm_history

    def plot_latent(self, y, y_noiseless=None):
        pred = self(y)
        mu, p, y_hat, sig, _ = pred
        fig, axes = plt.subplots(nrows=2, figsize=(12, 9))

        axes[0].plot(y_hat.squeeze().detach(), lw=2, label='Y_hat')
        axes[0].plot(y.squeeze().detach(), lw=1, label='Y_data')
        if y_noiseless is not None:
            axes[0].plot(y_noiseless[0].squeeze().detach(), lw=1, linestyle='--', label='Y_noiseless')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Magnitude')
        axes[0].grid(True)

        axes[1].plot(mu.squeeze().detach(), label='X_hat')
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Magnitude')
        axes[1].grid(True)

        [ax.legend() for ax in axes]
        plt.tight_layout()
        self.print_info()
        # print(f'B {self.mat_b}, R_chtril = {self.mat_r_chtril}')

        plt.show()

    def print_info(self):
        print(f'B {self.mat_b},\nR_chtril = {self.mat_r_chtril}\n------------------------------------------')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    torch.manual_seed(3)
    N_epochs = 100
    batch_size = 300

    sample_length = 100
    N_samples = 200
    N_states = 12

    # print('Test serializer.')
    # dataset_kwargs = {'usecols': ['IW_508']}
    # dataset_root = r'D:\DATA_FAST\EnergieMB_refined\trainingSet_cutEdgesScaledkW\Train'
    # motifs_serializer = DatasetSerializerCSV(dataset_root)
    # datasets = motifs_serializer.load_dataset(**dataset_kwargs)

    # for dataset in datasets:
    #     motifs_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    #     for obs in motifs_dataloader:
    #         print(f'sample.size: {obs[0].size()}')

    # Dummy data to fit
    time_vector = torch.linspace(0, 3, sample_length).unsqueeze(0)
    x_tensor = time_vector.repeat(N_samples, 1)
    y_noiseless = torch.stack([
        torch.sin(7 * x_tensor) + torch.cos(3 * x_tensor),
        torch.cos(5 * x_tensor)
    ],
                              dim=2).unsqueeze(-1)

    # y_noiseless = torch.tensor([[1., 1.]]) @ y_noiseless
    # y_noiseless = torch.tensor([[1., 0.], [0.1, 0.5], [-0.2, 0.5]]) @ y_noiseless
    y = y_noiseless + (0.15 * torch.randn(y_noiseless.size()))
    y_sample = y[0].unsqueeze(0)
    N_outputs = y.size()[-2]

    my_model = Model_hmm(sample_length, N_states, N_outputs)

    # my_model.plot_latent(y_sample, y_noiseless)

    my_model.print_info()
    res = my_model.fit_em(y, n_epochs=N_epochs, batch_size=batch_size, progressbar=True)
    # res = my_model.fit_bp(y, n_epochs=N_epochs, batch_size=batch_size, lr=0.02, progressbar=True)
    data_negloglikmean_history, frob_norm_sum_history = res
    my_model.print_info()

    # my_model.plot_latent(y_sample, y_noiseless)

    pred = my_model(y_sample)
    pred_bwd = my_model.roll_bwd(y_sample)
    ksi, gamma = pred_bwd
    mu, p, y_hat, sig, h = pred
    data_negloglikelihood = -my_model.data_meanlikelihood(y_sample, y_hat, sig)
    print(f'Test: neg. log-likelihood {data_negloglikelihood}')

    fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
    (ax0, ax1, ax2) = axes

    axes[0].plot(y_hat.squeeze().detach(), lw=2, label='Y_hat')
    axes[0].plot(y_sample.squeeze().detach(), lw=1, label='Y_data')
    axes[0].plot(y_noiseless[0].squeeze().detach(), lw=1, linestyle='--', label='Y_noiseless')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Magnitude')
    axes[0].grid(True)

    axes[1].plot(mu.squeeze().detach(), label='X_hat')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Magnitude')
    axes[1].grid(True)

    axes[2].plot(frob_norm_sum_history, label='Frob. norm sum history')
    axes[2].plot(data_negloglikmean_history, label='Mean negloglik history')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Magnitude')
    axes[2].grid(True)

    [ax.legend() for ax in axes]
    plt.tight_layout()
    plt.show()

    # ax.savefig(filenames[0] + '.pdf', format='pdf')
else:
    print('Importing data_serializer.py serializer')
