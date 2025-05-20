# Copyright (C) 2023-2025 J. Heinzmann, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import math
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg

from ..constraints import dirichlet_bcs
from . import assemble


# ===================================================================================
def predict(
    model,
    Fext: np.ndarray,
    u: np.ndarray,
    λ: np.ndarray,
    u_old: np.ndarray,
    λ_old: np.ndarray,
    s_bar: float,
):
    """returns predictor step solutions for arclength control method"""

    # assemble global system
    (_, Ktan) = assemble.assemble(model, u)

    # apply Dirichlet BCs
    (Ktan, _) = dirichlet_bcs.apply(model, Ktan, np.zeros(model.num_dofs))

    Δu_pred = sparse_linalg.spsolve(Ktan.tocsr(), Fext)

    # scale λ and determine current stiffness parameter
    Δλ = s_bar / math.sqrt(Δu_pred.T @ Δu_pred + 1)
    if np.all(u == u_old):
        # very first increment of computations (to prevent 0/0)
        current_stiffness = 0
    else:
        # from second increment onwards
        κ = Fext.T @ Δu_pred
        Δu = u - u_old
        κ_old = Fext.T @ Δu
        current_stiffness = κ / κ_old

    # store results of previous increment (n-1)
    u_old = u
    λ_old = λ

    # update load and displacement increments for predictor solution
    # depending on current stiffness parameter, choose correct path following direction
    if current_stiffness >= 0:
        λ = λ + Δλ
        u = u + Δu_pred * Δλ
    else:
        λ = λ - Δλ
        u = u - Δu_pred * Δλ

    return λ, λ_old, u, u_old


# ===================================================================================
def solve(
    Ktan: np.ndarray,
    Fext: np.ndarray,
    R: np.ndarray,
    u: np.ndarray,
    u_old: np.ndarray,
    λ: np.ndarray,
    λ_old: np.ndarray,
    s_bar: float,
):
    """
    solve system equations with constraint equation
    f = λ - λ_bar
    """

    # solution of the system with prior LU-decomposition of stiffness matrix for faster solving
    P, L, U = linalg.lu(Ktan.toarray(), check_finite=False)
    ΔU_λ = linalg.solve_triangular(
        U,
        linalg.solve_triangular(L, P.T @ Fext, lower=True, check_finite=False),
        check_finite=False,
    )
    ΔU_U = -linalg.solve_triangular(
        U,
        linalg.solve_triangular(L, P.T @ R, lower=True, check_finite=False),
        check_finite=False,
    )

    # computation of path following system quantities
    Δu = u - u_old
    s = math.sqrt(Δu.T @ Δu + (λ - λ_old) ** 2)
    f_ = s - s_bar
    K_λu = Δu.T / s
    K_λλ = (λ - λ_old) / s

    Δλ = -(f_ + K_λu @ ΔU_U) / (K_λλ + K_λu @ ΔU_λ)
    ΔU = ΔU_U + Δλ * ΔU_λ

    return ΔU, Δλ
