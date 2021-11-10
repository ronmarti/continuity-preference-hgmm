"""https://github.com/cornellius-gp/gpytorch/blob/92b3278eeb1ec93219f2a3921742a10aba9c8ceb/gpytorch/utils/cholesky.py#L11"""
import warnings

import torch

from .errors import NanError, NotPSDError


def psd_safe_cholesky(A, upper=False, out=None, jitter=None, max_tries=10):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
        :attr:`max_tries` (int, optional):
            Number of attempts (with successively increasing jitter) to make before raising an error.
    """
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(max_tries):
            jitter_new = jitter * (10**i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                # warnings.warn(f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal", NumericalWarning)
                return L
            except RuntimeError:
                continue
        raise RuntimeError(
            f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}. "
            f"Original error on first attempt: {e}")


def per_batch_cholesky(A, upper=False, out=None, jitter=None, max_tries=10):
    """Regularizes by jitter only ill-conditioned matrices from given batch
    """
    try:
        if A.dim() <= 2:
            return psd_safe_cholesky(A, upper, out, jitter, max_tries)
        results_list = []
        for batch_idx in range(A.shape[0]):
            chol_res = psd_safe_cholesky(A[batch_idx],
                                        upper=upper,
                                        out=out,
                                        jitter=jitter,
                                        max_tries=max_tries)
            results_list.append(chol_res)
        return torch.stack(results_list, dim=0)
    except RuntimeError as err:
        raise err


if __name__ == "__main__":
    a_root = torch.tensor([[[1, 0, 0], [-4, 0.7, 0], [1, 2, 0.5]],
                           [[10, 0, 0], [-1, 2, 0], [11, 2, 0]],
                           [[0.001, 0, 0], [-1, 2, 0], [1, 2, 1e-8]]])
    a_batch = a_root @ a_root.permute(0, 2, 1)
    print(per_batch_cholesky(a_batch))

    a_root = torch.tensor([[1, 0, 0], [-4, 0.7, 0], [1, 2, 0.5]])
    a_batch = a_root @ a_root.T
    print(per_batch_cholesky(a_batch))

else:
    print('Importing cholesky.py for psd safe cholesky')
