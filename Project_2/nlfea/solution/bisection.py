# Copyright (C) 2023-2025 J. Heinzmann, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np

from .. import constraints
from .. import solution


def iterate(
    model, solver, Fext, u, λ, u_old, λ_old, detK_old, detK_ref, s_bar, norm_ref
):
    """returns solution quantities for a increment for a critical point using bisection"""

    print("critical point")

    u_buffer = u
    u_old_buffer = u_old
    λ_buffer = λ
    λ_old_buffer = λ_old

    bisection_iteration = 0
    while bisection_iteration < solver.max_bisection_iter:
        # half the step size
        s_bar = s_bar / 2

        # predictor step
        (λ, λ_old, u, u_old) = solution.arclength.predict(
            model, Fext, u, λ, u_old, λ_old, s_bar
        )

        # iterative solution of increment
        (λ, u, stability, detK, res_norms, _, _) = (
            solution.newton_raphson.solve_increment(
                model, solver, Fext, u, λ, u_old, λ_old, s_bar, norm_ref
            )
        )

        # check convergence criteria
        print(
            f"\t\t\tbisection iteration: {bisection_iteration}: detK/detK_ref={detK / detK_ref}\n"
        )
        if ...: # TODO
            u = determine_critical_point_type(model, solver, Fext, u, λ)
            break

        # check if maximum number of iterations has been exceeded
        if bisection_iteration == (solver.max_bisection_iter - 1):
            raise AssertionError(
                f"Max. number of bisection iterations ({solver.max_bisection_iter}) exceeded!"
            )

        # if change of sign of detK, continue iteration from previous solution
        if np.sign(detK) != np.sign(detK_old):
            ... # TODO

        # if no change of sign of detK, continue iteration from current solution
        else:
            ... # TODO

        # increase bisection iteration counter
        bisection_iteration += 1

    return λ, λ_old, u, u_old, stability, detK, res_norms


# ===================================================================================
def determine_critical_point_type(model, solver, Fext, u, λ):
    """determine type of critical point and if requested return perturbed displacement vector"""

    # assembly tangent stiffness matrix
    (_, Ktan) = solution.assemble.assemble(model, u)
    Ktan, _ = constraints.dirichlet_bcs.apply(model, Ktan, np.zeros(model.num_dofs))

    # compute eigenvalues and eigenvectors
    (eigenvalues, eigenvectors) = np.linalg.eig(Ktan.toarray())

    # relaxed criterion on maximum value for zero-eigenvalues
    singular_entries = np.where(np.abs(eigenvalues) < 100 * solver.tolerance_detK)

    # loop over zero eigenvalues
    for singular_entry in np.nditer(singular_entries):
        # relaxed orthogonality requirement between force and eigenvector
        if (
            np.abs(eigenvectors[:, singular_entry].T @ Fext) / np.linalg.norm(Fext)
            < 10 * solver.tolerance_detK
        ):
            if solver.perturbation:
                u = u - 100 * solver.tolerance_detK * eigenvectors[:, singular_entry]

            print("bifurcation point")
        else:
            print("limit point")

    return u
