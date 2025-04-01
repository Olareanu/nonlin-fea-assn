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

    Δu_pred = ... # TODO

    # scale λ and determine current stiffness parameter
    Δλ = ... # TODO
    if np.all(u == u_old):
        # very first increment of computations (to prevent 0/0)
        current_stiffness = 0
    else:
        # from second increment onwards
        ... # TODO

    # store results of previous increment (n-1)
    u_old = u
    λ_old = λ

    # update load and displacement increments for predictor solution
    # depending on current stiffness parameter, choose correct path following direction
    if current_stiffness >= 0:
        ... # TODO
    else:
        ... # TODO

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
    ΔU_λ = ... # TODO
    ΔU_U = ... # TODO

    # computation of path following system quantities
    Δu = ... # TODO
    s = ... # TODO
    f_ = ... # TODO
    K_λu = ... # TODO
    K_λλ = ... # TODO

    Δλ = ... # TODO
    ΔU = ... # TODO

    return ΔU, Δλ
