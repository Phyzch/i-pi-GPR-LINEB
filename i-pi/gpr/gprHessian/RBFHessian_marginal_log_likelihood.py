import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
import torch
from torch import Tensor
import math
import time 
from contextlib import contextmanager

@contextmanager
def timer(name="Code"):
    start = time.perf_counter()
    
    yield
    
    elapsed = time.perf_counter() - start
    print(f"{name} took {elapsed / 60:.4f} min ({elapsed:.2f} s)")

class CustomMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """
    simple extension of ExactMarginalLogLikelihood class.
    We change the __forward__() function since num_data is different for our case (not function_dist.event_shape.numel() in ExactMarginalLogLikelihood)
    """

    def __init__(self, likelihood, model):
        super(CustomMarginalLogLikelihood, self).__init__(likelihood, model)

    def forward(self, function_dist, target, *params, 
                include_hessian= True, 
                compute_cond_number_bool= False,
                timing= False):
        """
        adopted from forward function in ExactMarginalLogLikelihood
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError(
                "ExactMarginalLogLikelihood can only operate on Gaussian random variables"
            )

        train_inputs = self.model.train_inputs[0]
        ndofs = train_inputs.shape[-1]
        train_size = train_inputs.shape[-2]
        func_grad_size = train_size * (1 + ndofs)  # size for potential and gradient part.

        if include_hessian:
            likelihood_kwargs = {}
        else:
            mean = function_dist.loc

            assert mean.shape[-1] == func_grad_size, (
                "The function distribution's size is inconsistent with the "
                "training data size when include_hessian = False"
            )
            assert target.shape[-1] == func_grad_size, (
                "The target's size is inconsistent with the training data size "
                "when include_hessian = False"
            )

            likelihood_kwargs = {"include_hessian": False}

        if timing:
            with timer("construct output:"):
                output = self.likelihood(function_dist, *params, **likelihood_kwargs)

            with timer("log prob"):
                res = output.log_prob(target)
        else:
            output = self.likelihood(function_dist, *params, **likelihood_kwargs)
            res = output.log_prob(target)

        if compute_cond_number_bool:
            self.compute_preconditioned_condition_number(output)

        res = self._add_other_terms(
            res, params
        )  # add penality term from prior distribution of parameters. (See gpytorch.mlls.exact_marginal_log_likelihood.py)

        # Scale by the amount of data we have
        res = res.div(train_size)

        return res


    def compute_preconditioned_condition_number(self, output: MultivariateNormal):
        """
        compute the condition number after preconditioning
        """
        covar = output.lazy_covariance_matrix
        covar = covar.evaluate_kernel() 
        # precond_lt is the linear operator representation of preconditioner 
        _, precond_lt, _ = covar._preconditioner() 

        covar_matrix = covar.to_dense()
        precond_matrix = precond_lt.to_dense()

        # solve generalized eigenvalues to obtain the condition number after preconditioning.
        largest_eigval, _ = torch.lobpcg(covar_matrix, k=1, B= precond_matrix, largest= True)
        smallest_eigval, _ = torch.lobpcg(covar_matrix, k=1, B= precond_matrix, largest= False)

        cond_number = largest_eigval[0] / smallest_eigval[0]
        pivoted_cholesky_num = gpytorch.settings.max_preconditioner_size.value()

        # we can also directly compute the condition number without preconditioning
        largest_eigval_no_precond, _ = torch.lobpcg(covar_matrix, k=1,  largest= True)
        smallest_eigval_no_precond, _ = torch.lobpcg(covar_matrix, k = 1, largest= False)

        cond_number_no_precond = largest_eigval_no_precond[0] / smallest_eigval_no_precond[0]

        print(f"condition number of original kernel matrix {cond_number_no_precond}")
        print(f"condition number after preconditioning: {cond_number}. pivoted Cholesky number {pivoted_cholesky_num}")