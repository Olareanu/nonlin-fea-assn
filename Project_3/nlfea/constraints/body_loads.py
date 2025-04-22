# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


# ===================================================================================
def apply(model, Fext: np.ndarray):
    """returns the external force vector with the contributions of the external body force"""

    # loop over tuple with all BC definitions
    for body_load in model.body_loads:
        # loop over all elements
        for element in model.elements:
            # compute contribution of current element to external force vector
            Fext_elem = model.parent_element.Fext_bodyload(element, body_load)

            # assemble element contributions at corresponding DOFs
            Fext[element.dofs] += Fext_elem

    return Fext
