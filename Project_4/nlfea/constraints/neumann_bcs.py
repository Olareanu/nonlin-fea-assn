# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


# ===================================================================================
def apply(model, Fext: np.ndarray):
    """
    returns the external force vector with the contributions of the external body force

    convention for definition of Neumann BCs:
    tuple containing tuples with:
    ("type", nodes, value)

    type:   either of T_X, T_Y, T_N (T_Z not implemented as this code is for 2D continuum elements)
    nodes:  list containing the indices of the nodes where the Neumann BC shall be applied
    value:  traction load

    for example:
    model.neumann_bcs = (("T_X", top_nodes, -0.1), )
    """

    # determine number of nodes the element type should have per edge
    if model.parent_element.element_type == "quad4":
        num_nodes_per_edge = 2
    elif (model.parent_element.element_type == "quad8") or (
        model.parent_element.element_type == "quad9"
    ):
        num_nodes_per_edge = 3

    # loop over tuple with all BC definitions
    for neumann_bc in model.neumann_bcs:
        # get the necessary information of Neumann BC
        direction = neumann_bc[0]
        nodes = neumann_bc[1]
        traction_load = neumann_bc[2]

        # loop over all elements
        for element in model.elements:
            # check whether current element has edge with traction load
            edge_nodes = np.isin(element.nodes, nodes, assume_unique=True)
            if sum(edge_nodes) == num_nodes_per_edge:
                # compute contribution of current element to external force vector
                Fext_elem = model.parent_element.Fext_traction(
                    element, edge_nodes, traction_load, direction
                )

                # assemble element contributions at corresponding DOFs
                Fext[element.dofs] += Fext_elem

    return Fext
