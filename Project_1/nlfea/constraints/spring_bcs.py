# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


# ===================================================================================
def apply(model, Ktan: np.ndarray, Fint: np.ndarray, u: np.ndarray):
    """
    returns tangential stiffness matrix and internal force vector with spring BCs

    convention for definition of spring BCs:
    tuple containing tuples with:
    ("type", nodes, value)

    type:   either of k_X, k_Y, k_Z
    nodes:  list containing the indices of the nodes where the spring BC shall be applied
    value:  stiffness of the spring

    for example:
    model.spring_bcs = (("k_X", bottom_nodes, 100), )
    """

    # loop over tuple with all BC definitions
    for spring_bc in model.spring_bcs:
        # get local DOF
        if spring_bc[0] == "k_X":
            local_dof = 0
        elif spring_bc[0] == "k_Y":
            local_dof = 1
        elif spring_bc[0] == "k_Z":
            local_dof = 2
        else:
            raise ValueError(f"type {spring_bc[0]} for Spring BCs not known")
        # get node where the spring BC shall be applied
        nodes = spring_bc[1]

        # get spring stiffness
        stiffness = spring_bc[2]

        # loop over the nodes of the given nodeset and impose the spring
        for node in nodes:
            dof = int(node * model.dimension + local_dof)
            # add spring force in Fint
            Fint[dof] += u[dof] * stiffness
            # update tangent matrix
            Ktan[dof,dof] += stiffness

    return Fint, Ktan
