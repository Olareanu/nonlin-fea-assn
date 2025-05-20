# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


# ===================================================================================
def find(nodes, **kwargs):
    """finds nodesets based on coordinate specification"""

    # start with nodeset containing all nodes
    nodeset = list(range(nodes.shape[0]))

    # loop over all given coordinate specifications
    for coordinate_specification, value_specification in kwargs.items():
        # translate X,Y,Z to numpy index 0,1,2
        coordinate_idxs = {"X": 0, "Y": 1, "Z": 2}
        coordinate_idx = coordinate_idxs[coordinate_specification]

        # get nodes satisfying specification
        nodeset_coordinates = np.where(
            np.isclose(nodes[:, coordinate_idx], value_specification)
        )[0].tolist()

        # keep only nodes satisfying current coordinate specification
        nodeset = list(set(nodeset) & set(nodeset_coordinates))

    return nodeset
