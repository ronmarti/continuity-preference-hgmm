from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset


def get_default_axes(num_axis: Tuple = (1, 1), figsize=(16, 8), **kwargs):
    return plt.subplots(*num_axis, figsize=figsize, **kwargs)


def plot_confidence(time,
                    mus,
                    covs,
                    ax=None,
                    sigma_level=2,
                    plot_kwargs: dict = dict(),
                    fill_kwargs: dict = dict()):
    if ax is None:
        ax = plt

    if time is not torch.Tensor:
        time = torch.tensor(time)

    if covs is not torch.Tensor:
        covs = torch.tensor(covs)

    if mus is torch.Tensor:
        mus = mus.detach().numpy()

    if 'alpha' not in fill_kwargs:
        fill_kwargs['alpha'] = 0.5

    devs = covs.diagonal(dim1=-2, dim2=-1).sqrt().detach().numpy()
    y1, y2 = mus-sigma_level*devs, mus+sigma_level*devs
    for y1_k, y2_k in zip(y1.T, y2.T):
        ax.fill_between(time, y1_k, y2_k, **fill_kwargs)
    ax.plot(time, mus, **plot_kwargs)


def plot_stratified_dset(stratified_dset: Dict[str, Tuple[Dataset, Dataset]],
                         ax: plt.axis = None):
    num_classes = len(stratified_dset)
    if ax is None:
        fig, ax = get_default_axes((2, num_classes))
    ax[0, 0].set_ylabel('Training')
    ax[1, 0].set_ylabel('Validation')

    for idx, (lbl, data) in enumerate(stratified_dset.items()):
        tr_labels = [tr_lbl for _, _, tr_lbl in data[0]]
        val_labels = [val_lbl for _, _, val_lbl in data[1]]
        print(f'Number of unique labels: {len(np.unique(tr_labels + val_labels))}')
        for tr_time, tr_data, tr_lbl in data[0]:
            ax[0, idx].plot(tr_time, tr_data.squeeze(-1))

        for val_time, val_data, val_lbl in data[1]:
            ax[1, idx].plot(val_time, val_data.squeeze(-1))
            ax[1, idx].set_xlabel(lbl)
    plt.tight_layout()
    plt.show()