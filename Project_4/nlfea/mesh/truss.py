# Copyright (C) 2023-2025 J. Heinzmann, H. C. Hille, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import math


# ===================================================================================
def von_mises(B: float, H: float):
    """returns nodes and elements of a von Mises truss mesh"""

    # nodal positions
    # [node_x node_y, node_z], row = node ID
    nodes = np.array([[-B, 0.0], [0.0, H], [B, 0.0]])

    # node assignment local to global
    # [node_1 node_2], row = element ID
    connectivity = np.array([[0, 1], [1, 2]], dtype=int)

    return nodes, connectivity


# ===================================================================================
def von_mises_sym(B: float, H: float):
    """returns nodes and elements of half a von Mises truss mesh (exploiting symmetry)"""

    # nodal positions
    # [node_x node_y, node_z], row = node ID
    nodes = np.array([[B, H], [0.0, 0.0]])

    # node assignment local to global
    # [node_1 node_2], row = element ID
    connectivity = np.array([[0, 1]], dtype=int)

    return nodes, connectivity


# ===================================================================================
def hexapod():
    """returns nodes and elements of a hexapod truss mesh"""

    # nodal positions
    # [node_x node_y, node_z], row = node ID
    nodes = np.array(
        [
            [0.0, 50.0, 0.0],
            [-math.cos(math.radians(30)) * 50.0, 25.0, 0.0],
            [-math.cos(math.radians(30)) * 50.0, -25.0, 0.0],
            [0.0, -50.0, 0.0],
            [math.cos(math.radians(30)) * 50.0, -25.0, 0.0],
            [math.cos(math.radians(30)) * 50.0, 25.0, 0.0],
            [-12.5, math.sin(math.radians(60)) * 25.0, 6.216],
            [-25.0, 0.0, 6.216],
            [-12.5, -math.sin(math.radians(60)) * 25.0, 6.216],
            [12.5, -math.sin(math.radians(60)) * 25.0, 6.216],
            [25.0, 0.0, 6.216],
            [12.5, math.sin(math.radians(60)) * 25.0, 6.216],
            [0.0, 0.0, 8.216],
        ]
    )

    # node assignment local to global
    # [node_1 node_2], row = element ID
    connectivity = np.array(
        [
            [0, 6],
            [6, 11],
            [11, 0],
            [6, 1],
            [1, 7],
            [7, 6],
            [7, 2],
            [2, 8],
            [8, 7],
            [8, 3],
            [3, 9],
            [9, 8],
            [9, 4],
            [4, 10],
            [10, 9],
            [10, 5],
            [5, 11],
            [11, 10],
            [6, 12],
            [7, 12],
            [8, 12],
            [9, 12],
            [10, 12],
            [11, 12],
        ],
        dtype=int,
    )

    return nodes, connectivity


# ===================================================================================
def tent():
    """returns nodes and elements of a tent truss mesh (3D von Mises truss)"""

    # nodal positions
    # [node_x node_y, node_z], row = node ID
    nodes = np.array(
        [
            [2.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 5.0],
            [2.0, 2.0, 5.0],
            [0.0, 2.0, 0.0],
            [-2.0, 2.0, 5.0],
        ]
    )

    # node assignment local to global
    # [node_1 node_2], row = element ID
    connectivity = np.array(
        [
            [0, 1],
            [1, 2],
            [0, 4],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 4],
            [3, 4],
            [4, 5],
        ],
        dtype=int,
    )

    return nodes, connectivity
