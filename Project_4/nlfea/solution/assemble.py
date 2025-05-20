# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
from scipy.sparse import coo_array


# ===================================================================================
def assemble(model, u: np.ndarray):
    """returns Fint and Ktan for the quad element with Total Lagrange formulation"""

    # initialize internal force vector and stiffness matrix
    Fint = np.zeros(model.num_dofs)
    Ktan_vals = np.zeros_like(model.Ktan_rows)

    # loop over model.elements tuple
    for element in model.elements:
        # extract assigned material of the element
        material_elem = model.material[element.domain_id]

        # extract the current displacement of the element
        u_elem = u[element.dofs]

        # compute element contributions
        (Fint_elem, Ktan_elem) = model.parent_element.Fint_Ktan(
            element, material_elem, u_elem
        )

        # assemble element contributions at corresponding DOFs
        Fint[element.dofs] += Fint_elem
        Ktan_vals[element.Ktan_sparse_idx] = Ktan_elem.flatten()

    # convert to sparse matrix
    Ktan = coo_array(
        (Ktan_vals, (model.Ktan_rows, model.Ktan_cols)), shape=(model.num_dofs,) * 2
    ).tolil()

    return Fint, Ktan
