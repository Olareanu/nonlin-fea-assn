# Copyright (C) 2023-2025 J. Heinzmann, H. C. Hille, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
from scipy import linalg as scipy_linalg
from scipy.sparse import linalg as sparse_linalg

from ..constraints import dirichlet_bcs
from ..constraints import spring_bcs

from . import assemble
from . import arclength


# ===================================================================================
def solve_increment(
    model,
    solver,
    Fext: np.ndarray,
    u: np.ndarray,
    λ: np.ndarray,
    u_old: np.ndarray = 0,
    λ_old: np.ndarray = 0,
    s_bar: float = 0,
    norm_ref: float = 0,
):
    """returns solution quantities for a increment by means of iterative solution with the Newton-Raphson method"""

    # initialize all values
    iteration = 0
    res_norms = []
    detK = None
    stability = None

    while iteration < solver.max_iter:
        # assemble global system
        (Fint, Ktan) = assemble.assemble(model, u)

        # incorporate spring boundary conditions
        if model.spring_bcs is not None:
            (Fint, Ktan) = spring_bcs.apply(model, Ktan, Fint, u)

        # residual vector with scaled external force
        R = Fint - λ * Fext

        # modify equation system according to Dirichlet BCs
        (Ktan, R) = dirichlet_bcs.apply(model, Ktan, R)

        # normalize residual to remove scaling with external forces either by:
        # - first iteration residual
        # - externally supplied value (for NR in bisection loop)
        residual_norm = np.linalg.norm(R, 2)
        if (iteration == 0) and (norm_ref == 0):
            norm_ref = residual_norm
        residual_norm = residual_norm / norm_ref

        # append to list of residuals
        res_norms.append(residual_norm)

        print(f"\t\titeration: {iteration}, residual_norm={residual_norm}")

        # exit iteration if solver.tolerance_nr criterion fulfilled
        if residual_norm < solver.tolerance_nr:
            # compute determinant of stiffness matrix if requested
            if solver.stability_analysis:
                # try Cholesky decomposition which fails if Ktan not positive definite
                try:
                    # use decomposition to efficiently calculate detK
                    L = scipy_linalg.cholesky(Ktan.toarray(), check_finite=False)
                    detK = np.prod(np.diag(L)) ** 2
                    stability = True
                except np.linalg.LinAlgError:
                    detK = np.linalg.det(Ktan.toarray())
                    stability = False

            # finish increment, i.e. break out of loop
            break

        # check if maximum number of iterations has been exceeded
        if iteration == (solver.max_iter - 1):
            raise AssertionError(
                f"Max. number of iterations ({solver.max_iter}) exceeded!"
            )

        # solve equation system depending on the control method
        if solver.control_method == "arclength":
            (Δu, Δλ) = arclength.solve(Ktan, Fext, R, u, u_old, λ, λ_old, s_bar)

            # update load factor
            λ = λ + Δλ

        elif solver.control_method == "load" or solver.control_method == "displacement":
            Δu = -sparse_linalg.spsolve(Ktan.tocsr(), R)

        # update displacement
        u = u + Δu

        iteration += 1

    return λ, u, stability, detK, res_norms, norm_ref, Fint
