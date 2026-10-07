"""
.. module:: cosmolayer.cosmolayer.cosmosolver
   :synopsis: Solves the self-consistent equation for the segment activity coefficients.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from __future__ import annotations

from typing import Any

import torch
from torch.autograd.function import FunctionCtx

from .utils import log_matmul_exp

NEWTON_STEP_TOLERANCE = {torch.float32: 1e-4, torch.float64: 1e-10}
NEWTON_RESIDUAL_TOLERANCE = {torch.float32: 1e-5, torch.float64: 1e-12}


class CosmoSolver(torch.autograd.Function):
    r"""COSMO self-consistent equation solver.

    Solves the COSMO self-consistent equations for the logarithm of the activity
    coefficient vector, :math:`\ln \boldsymbol{\gamma}`, given the nonnegative
    probability distribution vector :math:`\mathbf{p}` and the reduced interaction
    energy matrix :math:`\mathbf{U}/(RT)`.

    The self-consistent equations are:

    .. math::

        \boldsymbol{\gamma}\circ \left(
            \mathbf{B} ({\mathbf p} \circ \boldsymbol{\gamma})
        \right) = t \mathbf{1},

    where :math:`\mathbf{B} = \exp(-\mathbf{U}/(RT))` is the matrix of Boltzmann
    factors, :math:`t=\mathbf{1}^T \mathbf{p}` is the sum of the probabilities, and
    :math:`\circ` represents an elementwise product.

    The solution satisfies
    :math:`\boldsymbol{\gamma}^\mathsf{T} \mathbf{M} \boldsymbol{\gamma} = t`,
    where :math:`\mathbf{M} = \mathbf{B} \circ (\mathbf{p}\mathbf{p}^T)`.

    .. note::
        Supports batching, i.e., if :math:`\mathbf{p}` and :math:`\mathbf{U}/(RT)`
        can have broadcastable leading dimensions, all computations are performed
        in a single vectorized operation.

    Parameters
    ----------
    p : torch.Tensor
        Segment-type probability distribution vector. Must be nonnegative.
        Shape: (..., num_segment_types).
    U_RT : torch.Tensor
        Reduced interaction energy matrix :math:`\mathbf{U}/(RT)`.
        Shape: (..., num_segment_types, num_segment_types).
    max_iter : int
        Maximum number of iterations.
    max_step : float, optional
        Largest allowed change of :math:`\ln \boldsymbol{\gamma}` per Newton step.
        Each step is rescaled as a whole, keeping its direction, so that its largest
        component does not exceed ``max_step``. Default is None (no cap).

    Returns
    -------
    log_gamma : torch.Tensor
        The logarithm of the segment activity coefficient vector.
        Shape: (..., num_segment_types).

    Raises
    ------
    RuntimeError
        If the Newton solver does not converge within ``max_iter`` iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from cosmolayer.cosmosac import Component, CosmoSac2002Model
    >>> from importlib.resources import files
    >>> cosmo_strings = [
    ...     (files("cosmolayer.data") / f"{species}.cosmo").read_text()
    ...     for species in ("C=C(N)O", "NCCO")
    ... ]
    >>> probabilities = [
    ...     CosmoSac2002Model.create_component(cosmo_string).probabilities
    ...     for cosmo_string in cosmo_strings
    ... ]
    >>> p = torch.stack(
    ...     [torch.tensor(prob, dtype=torch.float32) for prob in probabilities],
    ... ).requires_grad_(True)
    >>> U_RT = torch.tensor(
    ...     CosmoSac2002Model.create_interaction_matrices(298.15)[0],
    ...     dtype=torch.float32,
    ...     requires_grad=True,
    ... )
    >>> log_gamma, converged = CosmoSolver.apply(p, U_RT)
    >>> converged.all().item()
    True
    >>> log_gamma
    tensor([[-4.5...e+00, -4.0...e+00, ... -1.3...e+01],
            [-2.1...e+01, -1.9...e+01, ... -5.3...e+00]], grad_fn=<CosmoSolverBackward>)
    >>> loss = (2 * log_gamma).exp().sum()
    >>> loss.backward()
    >>> p.grad
    tensor([[ 2.1...e+02,  2.1...e+02, ... -7.4...e+05],
            [-6.6...e+02, -6.3...e+02, ...  7.4...e+02]])
    """

    @staticmethod
    def logspace_newton_solver(
        p: torch.Tensor,
        U_RT: torch.Tensor,
        max_iter: int,
        max_step: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Solve the COSMO self-consistent equations with a log-space Newton method.

        Iterates in log-space for numerical stability. Runs under ``torch.no_grad()``
        when called from :meth:`forward`, but can also be called directly with
        gradient tracking enabled to backpropagate through the Newton iterations
        themselves, as an alternative to the implicit differentiation performed by
        :meth:`backward`.

        Parameters
        ----------
        p : torch.Tensor
            Segment-type probability distribution vector. Must be nonnegative.
            Shape: (..., num_segment_types).
        U_RT : torch.Tensor
            Reduced interaction energy matrix :math:`\mathbf{U}/(RT)`.
            Shape: (..., num_segment_types, num_segment_types).
        max_iter : int
            Maximum number of Newton iterations.
        max_step : float, optional
            Largest allowed change of :math:`\ln \boldsymbol{\gamma}` per Newton
            step; each step is rescaled as a whole to respect it. Default is None
            (no cap).

        Returns
        -------
        log_gamma : torch.Tensor
            The logarithm of the segment activity coefficient vector.
            Shape: (..., num_segment_types, 1).
        converged : torch.Tensor
            Boolean tensor indicating whether the iteration converged within
            ``max_iter`` steps.
        """
        step_tol = NEWTON_STEP_TOLERANCE[p.dtype]
        resid_tol = NEWTON_RESIDUAL_TOLERANCE[p.dtype]
        log_t = p.sum(dim=-1, keepdim=True).log().unsqueeze(-1)
        log_A = p.log().unsqueeze(-2) - U_RT
        Id = torch.eye(log_A.shape[-1], dtype=log_A.dtype, device=log_A.device)
        log_gamma = -torch.logsumexp(log_A, dim=-1, keepdim=True) + 0.5 * log_t
        log_A_gamma = log_matmul_exp(log_A, log_gamma)
        f = log_gamma + log_A_gamma - log_t
        failed = torch.zeros(f.shape[:-2], dtype=torch.bool, device=f.device)
        for _ in range(max_iter):
            J = torch.exp(log_gamma.mT + log_A - log_A_gamma) + Id
            delta, info = torch.linalg.solve_ex(J, -f, check_errors=False)
            solved = info.eq(0) & torch.isfinite(delta).all(dim=(-2, -1))
            solved &= ~failed
            failed |= ~solved
            delta = torch.where(solved[..., None, None], delta, torch.zeros_like(delta))
            if max_step is not None:
                scale = max_step / delta.abs().amax(dim=(-2, -1), keepdim=True)
                delta = delta * scale.clamp(max=1.0)
            log_gamma = log_gamma + delta
            log_A_gamma = log_matmul_exp(log_A, log_gamma)
            f = log_gamma + log_A_gamma - log_t
            delta_norm = delta.abs().amax(dim=(-2, -1))
            f_norm = f.abs().amax(dim=(-2, -1))
            converged = ~failed & (delta_norm < step_tol) & (f_norm < resid_tol)
            if bool(converged.all()):
                break
        return log_gamma, converged

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        p: torch.Tensor,
        U_RT: torch.Tensor,
        max_iter: int = 100,
        max_step: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx_any: Any = ctx
        ctx_any.p_shape = tuple(p.shape)
        ctx_any.u_shape = tuple(U_RT.shape)

        if max_iter <= 0:
            raise ValueError("Maximum number of iterations must be positive")

        invalid = (p < 0).any() | (p == 0).all(dim=-1).any()
        if bool(invalid):
            raise ValueError("Segment-type probabilities are invalid")

        with torch.no_grad():
            log_gamma, converged = CosmoSolver.logspace_newton_solver(
                p, U_RT, max_iter=max_iter, max_step=max_step
            )
        ctx.save_for_backward(log_gamma, p, U_RT, converged)

        return log_gamma.squeeze(-1), converged

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        grad_log_gamma = grad_outputs[0] if grad_outputs else None
        if grad_log_gamma is None:
            return None, None, None, None

        ctx_any: Any = ctx
        log_gamma, p, U_RT, converged = ctx_any.saved_tensors
        n_types = log_gamma.shape[-2]
        batch_shape = log_gamma.shape[:-2]
        p_expanded = torch.broadcast_to(p, (*batch_shape, n_types))
        U_expanded = torch.broadcast_to(U_RT, (*batch_shape, n_types, n_types))
        upstream = grad_log_gamma.reshape(-1, n_types)
        log_gamma_flat = log_gamma.reshape(-1, n_types, 1)
        p_flat = p_expanded.reshape(-1, n_types)
        U_flat = U_expanded.reshape(-1, n_types, n_types)

        active = torch.broadcast_to(converged, batch_shape).reshape(-1).clone()
        active &= torch.isfinite(upstream).all(dim=-1)
        active &= torch.isfinite(log_gamma_flat).all(dim=(-2, -1))
        active &= torch.isfinite(p_flat).all(dim=-1)
        active &= torch.isfinite(U_flat).all(dim=(-2, -1))

        grad_p_expanded = torch.zeros_like(p_expanded)
        grad_U_expanded = torch.zeros_like(U_expanded)
        active_indices = active.nonzero(as_tuple=False).flatten()
        if active_indices.numel():
            p_active = p_flat[active_indices]
            U_active = U_flat[active_indices]
            log_gamma_active = log_gamma_flat[active_indices]
            log_A = p_active.log().unsqueeze(-2) - U_active
            log_A_gamma = log_matmul_exp(log_A, log_gamma_active)
            Id = torch.eye(n_types, dtype=log_A.dtype, device=log_A.device)
            J = torch.exp(log_gamma_active.mT + log_A - log_A_gamma) + Id
            v, info = torch.linalg.solve_ex(
                J.mT, upstream[active_indices].unsqueeze(-1), check_errors=False
            )
            solved = info.eq(0) & torch.isfinite(v).all(dim=(-2, -1))

            if bool(solved.any()):
                solved_indices = active_indices[solved]
                p_solved = p_active[solved]
                U_solved = U_active[solved]
                log_gamma_solved = log_gamma_active[solved]
                log_A_gamma_solved = log_A_gamma[solved]
                v_solved = v[solved]

                gamma = log_gamma_solved.exp()
                B = torch.exp(-U_solved)
                r = v_solved / log_A_gamma_solved.exp()
                t = p_solved.sum(dim=-1, keepdim=True)
                grad_p = -(gamma * (B.mT @ r)).squeeze(-1) + v_solved.sum(dim=-2) / t
                pg = p_solved * gamma.squeeze(-1)
                grad_U_RT = r * B * pg.unsqueeze(-2)

                finite_p = torch.isfinite(grad_p).all(dim=-1)
                finite_U = torch.isfinite(grad_U_RT).all(dim=(-2, -1))
                grad_p_expanded.reshape(-1, n_types)[solved_indices[finite_p]] = grad_p[
                    finite_p
                ]
                grad_U_expanded.reshape(-1, n_types, n_types)[
                    solved_indices[finite_U]
                ] = grad_U_RT[finite_U]

        grad_p_result = grad_p_expanded.sum_to_size(ctx_any.p_shape)
        grad_U_result = grad_U_expanded.sum_to_size(ctx_any.u_shape)

        return grad_p_result, grad_U_result, None, None
