# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


# ===================================================================================
def apply(model, Fext: np.ndarray):
    """
    returns external force vector with contributions of point loads

    convention for definition of point loads:
    tuple containing tuples with:
    ("type", nodes, value)

    type:   either of F_X, F_Y, F_Z
    nodes:  list containing the indices of the nodes where the point load shall be applied
    value:  external force

    for example:
    model.point_loads = (("F_X", top_nodes, 100), )
    """

    # loop over tuple with all BC definitions
    for point_load in model.point_loads:
        # get local DOF
        if point_load[0] == "F_X":
            local_dof = 0
        elif point_load[0] == "F_Y":
            local_dof = 1
        elif point_load[0] == "F_Z":
            local_dof = 2
        else:
            raise ValueError(f"type {point_load[0]} for point_loads not known")

        # get node where the point load shall be applied
        nodes = point_load[1]

        # get external force
        force = point_load[2]

        # loop over the nodes of the given nodeset and impose the spring
        for node in nodes:
            dof = int(node * model.dimension + local_dof)

            # spring force acting against direction of displacement (external spring forces are added to internal force vector to avoid scaling by load factor variable λ)
            Fext[dof] += force

    return Fext
