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
    
    #------------------------------------------------------

    # Slides Step 1
    Δu_λ = sparse_linalg.spsolve(Ktan, Fext)

    # Slides Step 2

    Δλ = s_bar/np.sqrt(np.linalg.norm(Δu_λ)**2 + 1)

    Δu = Δu_λ * Δλ

    # Slides Step 3
    k = Fext.T @ Δu
    k_0 = Fext.T @ (u - u_old)
    #------------------------------------------------------

    # scale λ and determine current stiffness parameter
    if np.all(u == u_old):
        # very first increment of computations (to prevent 0/0)
        current_stiffness = 0
    else:
        # from second increment onwards
        current_stiffness = k/k_0 #------------------------------------------------------


    # store results of previous increment (n-1)
    u_old = u
    λ_old = λ

    # update load and displacement increments for predictor solution
    # depending on current stiffness parameter, choose correct path following direction
    if current_stiffness >= 0:
        #------------------------------------------------------
        u = u_old + Δu
        λ = λ_old + Δλ
        #------------------------------------------------------
    else:
        #------------------------------------------------------
        u = u_old - Δu
        λ = λ_old - Δλ
        #------------------------------------------------------

    # return u,λ  is at n, and u_old is at 
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
    
    #------------------------------------------------------
    # As per slide nr 8:
    ΔU_λ = linalg.solve_triangular(U, linalg.solve_triangular(L,P.T @ Fext, lower=True))
    ΔU_U = -linalg.solve_triangular(U, linalg.solve_triangular(L,P.T @ R, lower=True))
    #------------------------------------------------------

    # computation of path following system quantities
    #------------------------------------------------------
    
    # Per Slide 4:
    Δu = u - u_old
    s = np.sqrt(np.linalg.norm(u - u_old)**2 + (λ - λ_old)**2)
    f_ = s - s_bar
    K_λu = (u - u_old).T / s
    K_λλ = (λ - λ_old) / s

    Δλ = -(f_ + K_λu @ ΔU_U)/(K_λλ + K_λu @ ΔU_λ)
    ΔU = ΔU_U + ΔU_λ * Δλ
    #------------------------------------------------------

    return ΔU, Δλ
