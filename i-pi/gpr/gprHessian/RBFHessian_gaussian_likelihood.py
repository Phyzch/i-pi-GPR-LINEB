"""
The class that compute marginalized probability distribution after consider noise in the data. 
In the code, we transform noise from Cartesian coordinate into internal coordinate.
Written by Chenghao Zhang & Niri Govind. Pacific Northwest National Laboratory (chenghao.zhang@pnnl.gov), 2024.
"""

from typing import Any, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase, Likelihood
from gpytorch.priors import Prior
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.distributions import base_distributions
from gpytorch.lazy import LazyEvaluatedKernelTensor
import gpytorch

from linear_operator.operators import LinearOperator, DiagLinearOperator, LowRankRootLinearOperator
import linear_operator
import copy 

import time 
from contextlib import contextmanager
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Sparse CSR tensor support is in beta state.*",
    category=UserWarning,
)

@contextmanager
def timer(name="Code"):
    start = time.perf_counter()
    
    yield
    
    elapsed = time.perf_counter() - start
    print(f"{name} took {elapsed / 60:.4f} min ({elapsed:.2f} s)")

class RBFHessianGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    Base class for Gaussian Likelihoods for data with hessian information.

    :param: ndof: number of degrees of freedom
    :param: hessian_triu_size: number of upper triangle part of hessian matrix.
    :param: batch_shape: shape of batch
    :param: pot_noise_prior: prior for potential noise
    :param: force_noise_prior: prior for force noise
    :param: hessian_noise_prior: prior for hessian noise.
    :param: pot_noise_constraint, force_noise_constraint, hessian_noise_constraint: constraint for pot, force and hessian data.
    :param: has_covar_factor: The noise covariance matrix is not diagonal. The covariance matrix is expressed as covar_factor * covar_factor.T.
    :param: noise_covar_factor_pot_grad_array: An array of covar factor for all data points. The covar factor here perform transformation of gradient noise from Cartesian coordinate into internal coordinate.
    :param: noise_covar_factor_with_hessian_array: An array of covar factor for data points with hessians. The covar factor here perform transformation of gradient & hessian noise from Cartesian coordinate into internal coordinate.
    :param: grad_covar_factor_rank: the rank of covariance factor for gradient component. (this is number of dimensions in Cartesian coordinate.)
    :param: hessian_covar_factor_rank: the rank of covariance factor for hessian component. (this is number of upper triangle components in Cartesian coordinate.)
    """

    def __init__(
        self,
        ndof: int,
        hessian_triu_size: int,
        batch_shape: torch.Size = torch.Size(),
        pot_noise_prior: Optional[Prior] = None,
        pot_noise_constraint: Optional[Interval] = None,
        force_noise_prior: Optional[Prior] = None,
        force_noise_constraint: Optional[Interval] = None,
        hessian_noise_prior: Optional[Prior] = None,
        hessian_noise_constraint: Optional[Interval] = None,
        has_covar_factor: bool = False,
        noise_covar_factor_pot_grad_array: torch.Tensor = torch.tensor([]),
        noise_covar_factor_with_hessian_array: torch.Tensor = torch.tensor([]),
        grad_covar_factor_rank: int = 0,
        hessian_covar_factor_rank: int = 0,
        nugget: float = 0
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super(Likelihood, self).__init__()
        if pot_noise_constraint is None:
            pot_noise_constraint = GreaterThan(1e-8)

        if force_noise_constraint is None:
            force_noise_constraint = GreaterThan(1e-6)

        if hessian_noise_constraint is None:
            hessian_noise_constraint = GreaterThan(1e-4)

        self.batch_shape = batch_shape
        self.ndof = ndof
        self.hessian_triu_size = hessian_triu_size
        self.has_covar_factor = has_covar_factor
        self.nugget = nugget

        if not has_covar_factor:
            # The case that the noise of potential, gradient & hessian are (sigma_V)^2 I, (sigma_g)^2 I, (sigma_H)^2 I
            pass
        else:
            # the case that the covariance matrix of the noise of gradient & hessian is the product of covariance factor:  covar_factor @ covar_factor ^T
            if grad_covar_factor_rank == 0 or hessian_covar_factor_rank == 0:
                raise (
                    "Must provides the rank of covariance factor for gradient noise & hessian noise if the noise have covariant factor."
                )
            self.grad_covar_factor_rank = grad_covar_factor_rank
            self.hessian_covar_factor_rank = hessian_covar_factor_rank
            self.rank = (
                1 + grad_covar_factor_rank + hessian_covar_factor_rank
            )  # total rank (number of columns) for noise matrix of pot + grad + hessian.

            # check the shape of covar_factor_matrix
            self.noise_covar_factor_pot_grad_array = noise_covar_factor_pot_grad_array
            self.noise_covar_factor_with_hessian_array = (
                noise_covar_factor_with_hessian_array
            )
            if len(noise_covar_factor_pot_grad_array) != 0:
                row_size = noise_covar_factor_pot_grad_array[0].shape[0]
                column_size = noise_covar_factor_pot_grad_array[0].shape[1]
                assert (
                    row_size == 1 + ndof
                ), "the number of rows for covar_factor_pot_grad should be the same as the size of gradient in gpr model"
                assert (
                    column_size == 1 + grad_covar_factor_rank
                ), "the number of columns for covar_factor_pot_grad should be the same as the rank of gradient."

            if len(noise_covar_factor_with_hessian_array) != 0:
                row_size = noise_covar_factor_with_hessian_array[0].shape[0]
                column_size = noise_covar_factor_with_hessian_array[0].shape[1]
                assert (
                    row_size == 1 + ndof + hessian_triu_size
                ), "the number of rows for covar_factor_with_hessian should be the same as the size of gradient & hessian in gpr model"
                assert (
                    column_size
                    == 1 + grad_covar_factor_rank + hessian_covar_factor_rank
                ), "the number of columns for covar_factor_with_hessian should be the same as the rank of gradient & hessian."

        self.cached_noise_covar_factor = None
        self.cached_noise_covar_factor_M = None
        self.cached_noise_covar_factor_hessian_indices = None

        # register potential noises, the constraint for the potential noise & the prior for the potential noise
        # follwoing the convention in gpytorch, here pot_noises are variances of noise
        self.register_parameter(
            name="raw_pot_noises",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1, device= self.device)),
        )
        self.register_constraint("raw_pot_noises", pot_noise_constraint)
        if pot_noise_prior is not None:
            self.register_prior(
                "raw_pot_noises_prior", pot_noise_prior, lambda m: m.pot_noises
            )

        # register force noise, the constraint for the force noise & the prior for the force noise
        # following the convention in gpytorch, here force_noises are variances of noise
        self.register_parameter(
            name="raw_force_noises",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1, device= self.device)),
        )
        self.register_constraint("raw_force_noises", force_noise_constraint)
        if force_noise_prior is not None:
            self.register_prior(
                "raw_force_noises_prior", force_noise_prior, lambda m: m.force_noises
            )

        # register hessian noises, the constraint for the hessian noise and the prior for the hessian noise
        # following the convention in gpytorch, here hessian_noises are variances of noise
        self.register_parameter(
            name="raw_hessian_noises",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1, device= self.device)),
        )
        self.register_constraint("raw_hessian_noises", hessian_noise_constraint)
        if hessian_noise_prior is not None:
            self.register_prior(
                "raw_hessian_noises_prior",
                hessian_noise_prior,
                lambda m: m.hessian_noises,
            )

        self.cached_noise_covar = None 

    @property
    def pot_noises(self) -> Optional[Tensor]:
        # variance of potential noise.
        return self.raw_pot_noises_constraint.transform(self.raw_pot_noises)

    @pot_noises.setter
    def pot_noises(self, value: Union[float, Tensor]) -> None:
        self.initialize(
            raw_pot_noises=self.raw_pot_noises_constraint.inverse_transform(value)
        )

    @property
    def force_noises(self) -> Optional[Tensor]:
        # variance of force noises.
        return self.raw_force_noises_constraint.transform(self.raw_force_noises)

    @force_noises.setter
    def force_noises(self, value: Union[float, Tensor]) -> None:
        self.initialize(
            raw_force_noises=self.raw_force_noises_constraint.inverse_transform(value)
        )

    @property
    def hessian_noises(self) -> Optional[Tensor]:
        # variance of hessian noises.
        return self.raw_hessian_noises_constraint.transform(self.raw_hessian_noises)

    @hessian_noises.setter
    def hessian_noises(self, value: Union[float, Tensor]) -> None:
        self.initialize(
            raw_hessian_noises=self.raw_hessian_noises_constraint.inverse_transform(
                value
            )
        )

    def update_noise_covar_factor_array(
        self,
        new_noise_covar_factor_pot_grad_array: torch.Tensor,
        new_noise_covar_factor_with_hessian_array: torch.Tensor,
    ):
        """
        update the noise_covar_factor_array, which transform the noise from Cartesian coordinate into internal coordinate.
        Note here: for each data point, it has its own noise_covar_factor_pot_grad & noise_covar_factor_with_hessian (if the data point contains hessian information).
        Therefore, when we add new data into Gaussian Process Regression model, we need to update the noise_covar_factor array.
        :param: new_noise_covar_factor_pot_grad_array: the covar factor matrix for transformation of gradient noise for the new data point.
        :param: new_noise_covar_factor_with_hessian_array: the covar factor matrix for transformation of gradient + hessian noise for the new data point.
        """
        if len(new_noise_covar_factor_pot_grad_array) != 0:
            row_size = new_noise_covar_factor_pot_grad_array[0].shape[0]
            column_size = new_noise_covar_factor_pot_grad_array[0].shape[1]
            assert (
                row_size == 1 + self.ndof
            ), "the number of rows for covar_factor_pot_grad should fit the size of gradient in gpr model"
            assert (
                column_size == 1 + self.grad_covar_factor_rank
            ), "the number of columns for covar_factor_pot_grad should fit the rank of gradient."

        if len(new_noise_covar_factor_with_hessian_array) != 0:
            row_size = new_noise_covar_factor_with_hessian_array[0].shape[0]
            column_size = new_noise_covar_factor_with_hessian_array[0].shape[1]
            assert (
                row_size == 1 + self.ndof + self.hessian_triu_size
            ), "the number of rows for covar_factor_with_hessian should fit the size of gradient & hessian in gpr model"
            assert (
                column_size
                == 1 + self.grad_covar_factor_rank + self.hessian_covar_factor_rank
            ), "the number of columns for covar_factor_with_hessian should fit the rank of gradient & hessian."
        device = self.noise_covar_factor_pot_grad_array.device
        new_noise_covar_factor_pot_grad_array = new_noise_covar_factor_pot_grad_array.to(device= device)
        new_noise_covar_factor_with_hessian_array = new_noise_covar_factor_with_hessian_array.to(device= device)
        self.noise_covar_factor_pot_grad_array = torch.concat(
            [
                self.noise_covar_factor_pot_grad_array,
                new_noise_covar_factor_pot_grad_array,
            ]
        )
        self.noise_covar_factor_with_hessian_array = torch.concat(
            [
                self.noise_covar_factor_with_hessian_array,
                new_noise_covar_factor_with_hessian_array,
            ]
        )

        # clear the cache:
        self.cached_noise_covar_factor = None
        self.cached_noise_covar_factor_M = None
        self.cached_noise_covar_factor_hessian_indices = None

    def _shaped_noise_covar(
        self,
        M: int,
        hessian_data_point_index_array: torch.Tensor,
    ):
        """
        construct noise covar in block form.
        """
        device = self.device
        dtype = self.noise_covar_factor_pot_grad_array.dtype

        ndof = self.ndof
        grad_rank = self.grad_covar_factor_rank
        hess_rank = self.hessian_covar_factor_rank
        hess_size = self.hessian_triu_size

        hessian_idx = hessian_data_point_index_array.to(
            device=device,
            dtype=torch.long,
        )

        M_H = hessian_idx.numel()

        # ---------------------------------------------------------
        # Identify normal and Hessian data points
        # ---------------------------------------------------------
        all_idx = torch.arange(M, device=device)

        is_hessian = torch.zeros(
            M,
            dtype=torch.bool,
            device=device,
        )
        is_hessian[hessian_idx] = True

        normal_idx = all_idx[~is_hessian]
        N_normal = normal_idx.numel()

        # ---------------------------------------------------------
        # Noise standard deviations
        # ---------------------------------------------------------
        pot_std = torch.sqrt(
            self.pot_noises
        ).to(dtype=dtype, device=device)

        force_std = torch.sqrt(
            self.force_noises
        ).to(dtype=dtype, device=device)

        hessian_std = torch.sqrt(
            self.hessian_noises
        ).to(dtype=dtype, device=device)

        # =========================================================
        # 1. NORMAL DATA POINTS
        # =========================================================

        if N_normal > 0:

            # Shape:
            # [N_normal, 1 + ndof, 1 + grad_rank]
            blocks = self.noise_covar_factor_pot_grad_array[
                normal_idx
            ]

            # Column scaling:
            #
            # [potential column,
            #  gradient columns]
            #
            # shape = [1 + grad_rank]
            column_weights = torch.cat([
                pot_std.reshape(1),
                force_std.expand(grad_rank),
            ])

            weighted_blocks = (
                blocks * column_weights[None, None, :]
            )

            # -----------------------------------------------------
            # Block covariance
            #
            # [N_normal, 1+ndof, 1+ndof]
            # -----------------------------------------------------
            cov_blocks_normal = torch.bmm(
                weighted_blocks,
                weighted_blocks.transpose(1, 2),
            )

        else:
            cov_blocks_normal = None

        # =========================================================
        # 2. HESSIAN DATA POINTS
        # =========================================================

        if M_H > 0:

            # Shape:
            # [M_H,
            #  1 + ndof + hess_size,
            #  1 + grad_rank + hess_rank]
            blocks = (
                self.noise_covar_factor_with_hessian_array
            )

            # Column scaling:
            #
            # [potential,
            #  gradient,
            #  Hessian]
            #
            # shape = [1 + grad_rank + hess_rank]
            column_weights = torch.cat([
                pot_std.reshape(1),
                force_std.expand(grad_rank),
                hessian_std.expand(hess_rank),
            ])

            weighted_hessian_blocks = (
                blocks * column_weights[None, None, :]
            )

            # -----------------------------------------------------
            # Block covariance
            #
            # [M_H,
            #  1+ndof+hess_size,
            #  1+ndof+hess_size]
            # -----------------------------------------------------
            cov_blocks_hessian = torch.bmm(
                weighted_hessian_blocks,
                weighted_hessian_blocks.transpose(1, 2),
            )

        else:
            cov_blocks_hessian = None

        # =========================================================
        # 3. SCATTER BLOCKS BACK INTO GLOBAL MATRIX
        # =========================================================

        matrix_size = (
            M
            + M * ndof
            + M_H * hess_size
        )

        # We will build a sparse COO matrix directly.
        index_list = []
        value_list = []

        # ---------------------------------------------------------
        # Normal blocks
        # ---------------------------------------------------------

        if N_normal > 0:

            block_size = 1 + ndof

            # Global rows for each normal point:
            #
            # potential:
            #       i
            #
            # force:
            #       M + i*ndof + j
            #
            rows = torch.cat([
                normal_idx[:, None],

                M
                + normal_idx[:, None] * ndof
                + torch.arange(
                    ndof,
                    device=device,
                )[None, :],
            ], dim=1)

            # Expand row/column indices for the dense block
            block_rows = rows[:, :, None].expand(
                -1,
                block_size,
                block_size,
            )

            block_cols = rows[:, None, :].expand(
                -1,
                block_size,
                block_size,
            )

            index_list.append(
                torch.stack([
                    block_rows.reshape(-1),
                    block_cols.reshape(-1),
                ], dim=0)
            )

            value_list.append(
                cov_blocks_normal.reshape(-1)
            )

        # ---------------------------------------------------------
        # Hessian blocks
        # ---------------------------------------------------------

        if M_H > 0:

            block_size = 1 + ndof + hess_size

            hessian_position = torch.arange(
                M_H,
                device=device,
            )

            # Potential rows
            pot_rows = hessian_idx[:, None]

            # Force rows
            force_rows = (
                M
                + hessian_idx[:, None] * ndof
                + torch.arange(
                    ndof,
                    device=device,
                )[None, :]
            )

            # Hessian rows
            hessian_rows = (
                M
                + M * ndof
                + hessian_position[:, None] * hess_size
                + torch.arange(
                    hess_size,
                    device=device,
                )[None, :]
            )

            rows = torch.cat([
                pot_rows,
                force_rows,
                hessian_rows,
            ], dim=1)

            block_rows = rows[:, :, None].expand(
                -1,
                block_size,
                block_size,
            )

            block_cols = rows[:, None, :].expand(
                -1,
                block_size,
                block_size,
            )

            index_list.append(
                torch.stack([
                    block_rows.reshape(-1),
                    block_cols.reshape(-1),
                ], dim=0)
            )

            value_list.append(
                cov_blocks_hessian.reshape(-1)
            )

        # =========================================================
        # 4. CONSTRUCT GLOBAL SPARSE COVARIANCE
        # =========================================================

        indices = torch.cat(
            index_list,
            dim=1,
        )

        values = torch.cat(
            value_list
        )

        noise_covar_matrix = torch.sparse_coo_tensor(
            indices,
            values,
            size=(matrix_size, matrix_size),
            device=device,
            dtype=dtype,
        ).coalesce()

        return noise_covar_matrix

    def forward(self, function_samples: Tensor, *params, **kwargs: Any):
        """
        compute the conditional probability p(y|f(x)) of the tensor sample.
        :param: M: number of data points in the 1d data.
        :param: hessian_data_point_index_array: The index of data point that contains hessian information.
        """
        if len(params) != 2:
            print("Must pass M & M_H into likelihood function.")

        train_size = params[0]
        hessian_data_point_index_array = params[1]

        noise = self._shaped_noise_covar(train_size, hessian_data_point_index_array).to_dense().diagonal(
            dim1=-1, dim2=-2
        )  # take diagonal part of the matrix.

        include_hessian = kwargs.get('include_hessian', True)
        if include_hessian == False:
            # we do not include hessian part in the likelihood function. 
            pot_grad_size = train_size * (1 + self.ndof)
            noise = noise[..., :pot_grad_size, :pot_grad_size]

            assert(function_samples.shape[-1] == pot_grad_size), \
            "the size of function_samples should fit the size of potential and gradient part in gpr model if include_hessian is False."

        return base_distributions.Independent(
            base_distributions.Normal(function_samples, noise.sqrt()), 1
        )

    def marginal(
        self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> MultivariateNormal:
        r"""
        compute the marginal probability p(y|X), marginalize over f(X).
        the covariance matrix will be K + noise_covar_matrix.
        :param: function_dist: latent function distribution f(X). MultivariateNormal distribution.
        :param: M: number of data points in the 1d data.
        :param: hessian_data_point_index_array: The index of data point that contains hessian information.
        """
        if len(params) != 2:
            print("Must pass M & M_H into likelihood function.")

        train_size = params[0]
        hessian_data_point_index_array = params[1]
        # covar: covariance matrix of the latent function.
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        # ensure that sumKroneckerLT is actually called
        if isinstance(covar, LazyEvaluatedKernelTensor):
            covar = covar.evaluate_kernel()

        # compute the covariance matrix of the noise.
        noise_covar = self._shaped_noise_covar(train_size, hessian_data_point_index_array)
        noise_covar = noise_covar.to_dense()
        noise_covar = linear_operator.to_linear_operator(noise_covar)

        include_hessian = kwargs.get('include_hessian', True)
        if include_hessian == False:
            # we do not include hessian part in the likelihood function. 
            pot_grad_size = train_size * (1 + self.ndof)
            assert(covar.shape[-1] == pot_grad_size), \
            "the size of covariance matrix of latent function should fit the size of potential and gradient part in gpr model if include_hessian is False."
            noise_covar = noise_covar[..., :pot_grad_size, :pot_grad_size] 

        # compute the largest eigenval of covariance matrix.
        # It's important to add diagonal_nugget here, this will make full_covar as linear_operator.AddedDiagLinearOperator object,
        # which enables the preconditioned Cholesky decomposition as preconditioner .
        eigval_max = power_iteration(covar, num_iters= 20)
        diagonal_nugget = DiagLinearOperator(torch.ones(covar.shape[0])).to(device= covar.device) * self.nugget * eigval_max

        full_covar = covar + noise_covar + diagonal_nugget 

        return function_dist.__class__(mean, full_covar)
    
def power_iteration(mat: gpytorch.lazy.LazyEvaluatedKernelTensor, num_iters=20):
    """
    compute largest eigenvalue using Rayleigh quotient 
    """
    n = mat.shape[-1]
    b_k = torch.randn(n, device=mat.device, dtype= torch.float64)
    for _ in range(num_iters):
        b_k1 = mat @ b_k
        b_k1_norm = b_k1.norm()
        b_k = b_k1 / b_k1_norm
    return (b_k @ (mat @ b_k)).item() 