# Copyright (C) 2023-2025 J. Heinzmann, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import time

from .. import constraints
from .. import post
from .. import solution


# ===================================================================================
def solve(model, solver):
    """incremental solver"""

    # save time for runtime information
    time_start = time.time()

    # directly compute external force vector since it independent of displacement and will not change
    Fext = np.zeros(model.num_dofs)
    if model.neumann_bcs is not None:
        Fext = constraints.neumann_bcs.apply(model, Fext)
    if model.body_loads is not None:
        Fext = constraints.body_loads.apply(model, Fext)
    if model.point_loads is not None:
        Fext = constraints.point_loads.apply(model, Fext)

    # initialization of displacement vector and load factor
    u = np.zeros(model.num_dofs)
    u_old = u
    λ = 0
    λ_old = λ

    # ===================================================================================
    # incremental-iterative solution
    increment = 0
    while increment < solver.num_increments:
        print("_" * 75, f"\nincrement: {increment}")

        if solver.control_method == "load":
            # increment load factor by fixed control size
            λ += 1 / solver.num_increments

        elif solver.control_method == "displacement":
            # increment load factor by fixed control size
            λ += 1 / solver.num_increments

            # impose Dirichlet BCs for the current step
            u = constraints.dirichlet_bcs.impose(model, u, λ)

        elif solver.control_method == "arclength":
            # predictor step
            (λ, λ_old, u, u_old) = solution.arclength.predict(
                model, Fext, u, λ, u_old, λ_old, solver.control_size
            )

        # iterative solution of increment
        (λ, u, stability, detK, res_norms, norm_ref, Fint) = (
            solution.newton_raphson.solve_increment(
                model, solver, Fext, u, λ, u_old, λ_old, solver.control_size
            )
        )

        # check for sign change in detK (not at zeroth increment)
        if solver.stability_analysis and (increment == 0):
            (detK_ref, detK_old, u_nm1, λ_nm1) = (detK, detK, u_old, λ_old)

            solver.monitoring["critical"].append(False)

        # skip cases in which last increment was critical point due to large uncertainty in the sign of detK close to critical point
        elif solver.stability_analysis and (increment > 0):
            if (abs(detK_old / detK_ref) > solver.tolerance_detK) and (
                np.sign(detK) != np.sign(detK_old)
            ):
                # mark increment as critical point
                solver.monitoring["critical"].append(True)

                # perform bisection
                (λ, λ_old, u, u_old, stability, detK, res_norms) = (
                    solution.bisection.iterate(
                        model,
                        solver,
                        Fext,
                        u_nm1,
                        λ_nm1,
                        u_nm2,  # noqa: F821
                        λ_nm2,  # noqa: F821
                        detK_old,
                        detK_ref,
                        solver.control_size,
                        norm_ref,
                    )
                )
            else:
                solver.monitoring["critical"].append(False)

            # update old states
            (detK_old, u_nm2, u_nm1, λ_nm2, λ_nm1) = (detK, u_nm1, u, λ_nm1, λ)  # noqa: F841

        # store relevant results at monitoring DOF
        solver.monitoring["u"].append(u[solver.monitoring_dof])
        solver.monitoring["F"].append(Fint[solver.monitoring_dof])
        solver.monitoring["λ"].append(λ)
        solver.monitoring["detK"].append(detK)
        solver.monitoring["stability"].append(stability)
        solver.monitoring["residuals"].append(res_norms)

        # output fields
        post.output.write_vtk(
            solver.result_dir, increment, model, {"u": u, "Fint": Fint}
        )

        # check if end of load path is reached
        if λ > 1:
            break

        # increase increment number
        increment += 1

    # runtime information
    time_end = time.time()
    print(f"\nsolution finished after {time_end - time_start:.3f}s")
