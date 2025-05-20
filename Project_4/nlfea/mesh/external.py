# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import meshio


# ===================================================================================
def load(filepath: str):
    """
    returns nodes and elements from an external mesh file, relying on the module 'meshio'
    (https://pypi.org/project/meshio/, published under the MIT licence)

    Schlömer, N. meshio: Tools for mesh files [Computer software]
    https://doi.org/10.5281/zenodo.1173115
    """

    # read mesh
    input_mesh = meshio.read(filepath)

    # transform into internally expected format
    nodes = input_mesh.points
    connectivity = input_mesh.cells[0].data

    return nodes, connectivity
