"""A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk"""
import torch
import math
import numpy as np
# from scipy import linalg
from scipy.linalg import solve_sylvester


def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast.
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def vec(a):
    """Column-major vectorization. First dimension is batch dim.
    Keeps the singleton trailing dimension.
    """
    a_vec = a.permute(0, 2, 1).flatten(start_dim=1).unsqueeze(2)
    return a_vec


def de_vec(a, rows, cols):
    """Column-major de-vectorization. First dimension is batch dim.
    """
    batch = a.size(0)
    # dim = math.floor(math.sqrt(a.size(1)))
    a_mat = a.reshape(batch, cols, rows).permute(0, 2, 1)
    return a_mat


def sylvester(a, b, q):
    """Takes torch tensors and solves the sylvester equation for X:
    AX + XB = Q
    """
    if a.dim() == 3:
        assert a.size(0) == b.size(0) == q.size(0),\
            f'Batch sizes must equal, got a: {a.size(0)},\
                b: {b.size(0)}, c: {q.size(0)}.'
        results = []
        for i in range(a.size(0)):
            res = solve_sylvester(a[i].detach().numpy(), b[i].detach().numpy(), q[i].detach().numpy())
            results.append(torch.from_numpy(res))
        return torch.stack(results, 0)
    else:
        res = solve_sylvester(a.detach().numpy(), b.detach().numpy(), q.detach().numpy())
        return torch.from_numpy(res)


if __name__ == "__main__":
    a = torch.tensor([[1., 2.]])
    b = torch.tensor([[1., 3.], [0.1, 0.5]])
    print(f'Kronecker product a = {a},\nb = {b}\na kron b = {kron(a, b)}')

    c = torch.tensor([[[1, 2, 2.5], [3, 4, 4.5]], [[5, 6, 6.5], [7, 8, 8.5]]])
    c = c.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    print(f'c = {c}')
    d = vec(c).contiguous()
    print(f'vec(c) = {d},\nsize: {d.size()}')
    e = de_vec(d, 2, 3)
    print(f'de_vec(d) = {e},\nsize: {e.size()}')

    a_s = torch.tensor([[[3, 2.5], [6, 4.5]], [[3, 6.5], [7, 14]]])
    b_s = torch.tensor([[[1, 2, 2], [3, 4.5, 3], [1, 4.5, 1]], [[5, 6.5, 7], [7, 8.5, 1], [3, 5, 5]]])
    q_s = torch.tensor([[[1, 2], [3, 4.5], [1, 4.5]], [[5, 6.5], [7, 8.5], [3, 5]]]).permute(0, 2, 1)
    x = sylvester(a_s, b_s, q_s)
    print(f'sylvester: {x}, of size {x.size()}')
    print(f'ax + xb = q is {a_s @ x + x @ b_s} = {q_s}')
    syl_dif = a_s @ x + x @ b_s - q_s
    syl_dif_norm = torch.norm(syl_dif, p='fro', dim=(1, 2))
    print(f'sylvester frob norm of err: {syl_dif_norm}')
else:
    print('Importing kronecker.py')
