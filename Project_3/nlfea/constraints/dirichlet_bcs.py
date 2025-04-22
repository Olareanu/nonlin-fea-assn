# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


# ===================================================================================
def impose(model, u: np.ndarray, λ: float):
    """
    returns displacement vector with applied Dirichlet BCs

    convention for definition of Dirichlet BCs:
    tuple containing tuples with:
    ("type", nodes) or ("type", nodes, value)

    type:   either of fix_X, fix_Y, fix_Z, u_X, u_Y, u_Z
    nodes:  list containing the indices of the nodes where the Dirchlet BC shall be applied
    value:  either lambda function depending on λ or numerical value, necessary only for type = u_X, u_Y, u_Z

    for example:
    model.dirichlet_bcs = (("fix_X", bottom_nodes), ("u_Y", top_nodes, 2))
    """

    # loop over tuple with all BC definitions
    for dirichlet_bc in model.dirichlet_bcs:
        # get local DOF
        if dirichlet_bc[0] == "fix_X":
            local_dof = 0

        elif dirichlet_bc[0] == "fix_Y":
            local_dof = 1

        elif dirichlet_bc[0] == "fix_Z":
            local_dof = 2

        elif dirichlet_bc[0] == "u_X":
            local_dof = 0

        elif dirichlet_bc[0] == "u_Y":
            local_dof = 1

        elif dirichlet_bc[0] == "u_Z":
            local_dof = 2

        else:
            raise ValueError(f"type {dirichlet_bc[0]} for Dirichlet BCs not known")

        # get the nodes where Dirichlet BCs are to be applied
        nodes = dirichlet_bc[1]

        # get the value of the Dirichlet BC
        if dirichlet_bc[0] in ["fix_X", "fix_Y", "fix_Z"]:
            value = 0.0
        elif dirichlet_bc[0] in ["u_X", "u_Y", "u_Z"]:
            value = dirichlet_bc[2]

        # check if value is a lambda function
        if callable(value):
            displacement = value(λ)
        else:
            displacement = λ * value

        # loop over the nodes of the given nodeset and impose the value
        for node in nodes:
            dof = int(node * model.dimension + local_dof)
            u[dof] = displacement

    return u


# ===================================================================================
def apply(model, Ktan: np.ndarray, R: np.ndarray):
    """
    returns modified tangential stiffness matrix and residual vector for Dirichlet BCs
    the method essentially turns off the DOFs where Dirichlet BCs have been applied

    convention for definition of Dirichlet BCs:
    tuple containing tuples with:
    ("type", nodes) or ("type", nodes, value)

    type:   either of fix_X, fix_Y, fix_Z, u_X, u_Y, u_Z
    nodes:  list containing the indices of the nodes where the Dirchlet BC shall be applied
    value:  either lambda function depending on λ or numerical value, necessary only for type = u_X, u_Y, u_Z

    for example:
    model.dirichlet_bcs = (("fix_X", bottom_nodes), ("u_Y", top_nodes, 2))
    """

    for dirichlet_bc in model.dirichlet_bcs:
        # get the local DOF for the Dirichlet BC
        if dirichlet_bc[0] == "fix_X":
            local_dof = 0

        elif dirichlet_bc[0] == "fix_Y":
            local_dof = 1

        elif dirichlet_bc[0] == "fix_Z":
            local_dof = 2

        elif dirichlet_bc[0] == "u_X":
            local_dof = 0

        elif dirichlet_bc[0] == "u_Y":
            local_dof = 1

        elif dirichlet_bc[0] == "u_Z":
            local_dof = 2

        else:
            raise ValueError(f"type {dirichlet_bc[0]} for Dirichlet BCs not known")

        # get the nodes where Dirichlet BCs are to be applied
        nodes = dirichlet_bc[1]

        # loop over the nodes of the given nodeset and deactivate the respective DOFs
        for node in nodes:
            dof = int(node * model.dimension + local_dof)
            R[dof] = 0
            Ktan[dof, :] = 0
            Ktan[:, dof] = 0
            Ktan[dof, dof] = 1

    return Ktan, R
