# Copyright (C) 2023-2025 J. Heinzmann, H. C. Hille, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import math


# ===================================================================================
def patch_undistorted(W: float, H: float, element_type: str):
    """returns nodes and elements for a patch test mesh with undistorted elements"""

    # check for ansatz order
    if element_type == "quad4":
        num_nodes_edge = 3
    elif element_type == "quad8" or element_type == "quad9":
        num_nodes_edge = 5

    # get specific rectangle mesh of dimension (0,W) x (0,H)
    (nodes, elements) = rectangle(
        length_x=W,
        length_y=H,
        num_nodes_x=num_nodes_edge,
        num_nodes_y=num_nodes_edge,
        element_type=element_type,
    )

    return nodes, elements


# ===================================================================================
def patch_distorted(W: float, H: float, element_type: str):
    """returns nodes and elements for a patch test mesh with distorted elements"""

    # start with undistorted patch test mesh
    (nodes, elements) = patch_undistorted(W, H, element_type)

    # distort mesh by moving the nodes
    for node_idx in range(nodes.shape[0]):
        x = nodes[node_idx, 0]
        y = nodes[node_idx, 1]

        nodes[node_idx, 0] = x + 1 / 6 * W * (
            np.heaviside(-np.sign(x - W / 2), 1) * x / (W / 2)
            + np.heaviside(np.sign(x - 0.5), 0) * (2 - x / (W / 2))
        ) * (
            np.heaviside(-np.sign(y - H / 2), 1) * y / (H / 2)
            + np.heaviside(np.sign(y - 0.5), 0) * (2 - y / (H / 2))
        )

        nodes[node_idx, 1] = y + 1 / 6 * H * (
            np.heaviside(-np.sign(y - H / 2), 1) * y / (H / 2)
            + np.heaviside(np.sign(y - 0.5), 0) * (2 - y / (H / 2))
        ) * (
            np.heaviside(-np.sign(x - W / 2), 1) * x / (W / 2)
            + np.heaviside(np.sign(x - 0.5), 0) * (2 - x / (W / 2))
        )

    return nodes, elements


# ===================================================================================
def contact_patch(
    length_x: float,
    length_y: float,
    num_nodes_x: int,
    num_nodes_y: int,
    element_type: str,
):
    """returns nodes and elements for a contact patch test mesh with elements of arbitrary width"""

    # nodal positions (global node number, position coordinates)
    nodes = np.zeros((num_nodes_x * num_nodes_y, 2))
    node_map = np.zeros((num_nodes_x * num_nodes_y, 1))

    # initialize coordinates
    coords_x = np.linspace(0, length_x, num_nodes_x)
    # random perturbation of x coordinates by 0.9 times regular element width
    coords_x[1:-1] += (
        (np.random.rand(coords_x.shape[0] - 2) - 0.5)
        * 0.9
        * length_x
        / (num_nodes_x - 1)
    )
    coords_y = np.linspace(0, length_y, num_nodes_y)

    # assign nodal coordinates
    # [node_x node_y], row = node ID
    node_idx = 0
    internal_idx = 0
    for j in range(num_nodes_y):
        for i in range(num_nodes_x):
            # check wheter at internal node position
            if element_type == "quad8" and (((i + 1) % 2 == 0) and ((j + 1) % 2 == 0)):
                nodes[node_idx, :] = np.full((1, 2), np.nan)
                node_map[node_idx, 0] = np.nan
                internal_idx += 1
            else:
                nodes[node_idx, :] = np.array([[coords_x[i], coords_y[j]]])
                node_map[node_idx, 0] = node_idx - internal_idx

            node_idx += 1

    # create matrix of nodes for element connectivity
    node_matrix = np.reshape(node_map, (num_nodes_y, num_nodes_x)).T

    # delete nan rows from internal points (for serendipity elements)
    nodes = np.delete(nodes, np.isnan(nodes[:, 0]), 0)

    # obtain connectivity from helper function
    elements = connectivity(node_matrix, num_nodes_x, num_nodes_y, element_type)

    return nodes, elements


# ===================================================================================
def rectangle(
    length_x: float,
    length_y: float,
    num_nodes_x: int,
    num_nodes_y: int,
    element_type: str,
):
    """returns nodes and elements for a rectangular plane mesh (used as basis for some of the other meshes)"""

    # nodal positions (global node number, position coordinates)
    nodes = np.zeros((num_nodes_x * num_nodes_y, 2))
    node_map = np.zeros((num_nodes_x * num_nodes_y, 1))

    # initialize coordinates
    coords_x = np.linspace(0, length_x, num_nodes_x)
    coords_y = np.linspace(0, length_y, num_nodes_y)

    # assign nodal coordinates
    # [node_x node_y], row = node ID
    node_idx = 0
    internal_idx = 0
    for j in range(num_nodes_y):
        for i in range(num_nodes_x):
            # check wheter at internal node position
            if element_type == "quad8" and (((i + 1) % 2 == 0) and ((j + 1) % 2 == 0)):
                nodes[node_idx, :] = np.full((1, 2), np.nan)
                node_map[node_idx, 0] = np.nan
                internal_idx += 1
            else:
                nodes[node_idx, :] = np.array([[coords_x[i], coords_y[j]]])
                node_map[node_idx, 0] = node_idx - internal_idx

            node_idx += 1

    # create matrix of nodes for element connectivity
    node_matrix = np.reshape(node_map, (num_nodes_y, num_nodes_x)).T

    # delete nan rows from internal points (for serendipity elements)
    nodes = np.delete(nodes, np.isnan(nodes[:, 0]), 0)

    # obtain connectivity from helper function
    elements = connectivity(node_matrix, num_nodes_x, num_nodes_y, element_type)

    return nodes, elements


# ===================================================================================
def cooksmembrane(num_nodes_x: int, num_nodes_y: int, element_type: str):
    """
    returns nodes and elements for cooke's membrane example, based on:

    Schröder et. al (2021): "A selection of Benchmark Problems in Solid Mechanics and Applied Mathematics"
    https://doi.org/10.1007/s11831-020-09477-3, p. 715

    Cook (1974): "Improved Two-Dimensional Finite Element"
    https://doi.org/10.1061/JSDEAG.0003877
    """

    # start with rectangle mesh of dimension (0,1) x (0,1)
    (nodes, elements) = rectangle(
        length_x=1,
        length_y=1,
        num_nodes_x=num_nodes_x,
        num_nodes_y=num_nodes_y,
        element_type=element_type,
    )

    # transform mesh to target coordinates
    nodes[:, 1] = (44 - (44 - 16) * nodes[:, 0]) * nodes[:, 1] + 44 * nodes[:, 0]
    nodes[:, 0] = 48 * nodes[:, 0]

    return nodes, elements


# ===================================================================================
def arc(
    r_inner: float,
    r_outer: float,
    φ_start: float,
    φ_end: float,
    num_nodes_r: int,
    num_nodes_φ: int,
    element_type: str,
):
    """returns nodes and elements for an arc (segment) plane mesh"""

    # initialize nodal positions (global node number, position coordinates)
    nodes = np.zeros((num_nodes_r * num_nodes_φ, 2))
    node_map = np.zeros((num_nodes_r * num_nodes_φ, 1))

    # determine nodal positions
    r = np.linspace(r_inner, r_outer, num_nodes_r)
    φ = np.linspace(φ_start, φ_end, num_nodes_φ)

    # loop over all possible combinations of discrete r and φ coordinates
    node_idx = 0
    internal_idx = 0
    for j in range(num_nodes_φ):
        for i in range(num_nodes_r):
            # check wheter at internal node position
            if element_type == "quad8" and (((i + 1) % 2 == 0) and ((j + 1) % 2 == 0)):
                nodes[node_idx, :] = np.full((1, 2), np.nan)
                node_map[node_idx, 0] = np.nan
                internal_idx += 1
            else:
                nodes[node_idx, :] = [
                    r[i] * math.cos(np.deg2rad(φ[j])),
                    r[i] * math.sin(np.deg2rad(φ[j])),
                ]
                node_map[node_idx, 0] = node_idx - internal_idx

            node_idx += 1

    # create matrix of nodes for element connectivity
    node_matrix = np.reshape(node_map, (num_nodes_φ, num_nodes_r)).T

    # delete nan rows from internal points (for serendipity elements)
    nodes = np.delete(nodes, np.isnan(nodes[:, 0]), 0)

    # obtain connectivity from helper function
    elements = connectivity(node_matrix, num_nodes_r, num_nodes_φ, element_type)

    return nodes, elements


# ===================================================================================
def connectivity(node_matrix, num_nodes_dir1, num_nodes_dir2, element_type):
    """returns the element connectivity for a two-dimensional mesh based on the element (ansatz order and serendipity)"""

    # linear elements (quad4)
    if element_type == "quad4":
        if (num_nodes_dir1 < 2) or (num_nodes_dir2 < 2):
            raise ValueError(
                "elements with linear ansatz functions require >2 nodes in each direction!"
            )

        # initialize elements array
        elements = np.zeros(((num_nodes_dir1 - 1) * (num_nodes_dir2 - 1), 4), dtype=int)

        # loop over all elements
        elem_idx = 0
        for j in range(num_nodes_dir2 - 1):
            for i in range(num_nodes_dir1 - 1):
                elements[elem_idx, :] = np.array(
                    [
                        [
                            node_matrix[i, j],
                            node_matrix[i + 1, j],
                            node_matrix[i + 1, j + 1],
                            node_matrix[i, j + 1],
                        ]
                    ]
                )
                elem_idx += 1

    # quadratic elements
    if element_type == "quad8" or element_type == "quad9":
        if ((num_nodes_dir1 > 2) or (num_nodes_dir2 > 2)) and (
            (num_nodes_dir1 % 2 == 0) or (num_nodes_dir2 % 2 == 0)
        ):
            raise ValueError(
                "elements with quadratic ansatz functions require an odd number of elements in each direction"
            )

        # elements with internal nodes (quad9)
        if element_type == "quad9":
            # local node IDs with (ξ, η):
            #
            #              (0,1)
            #   (-1, 1) 3 ---6--- 2 (1, 1)
            #           |         |
            #    (-1,0) 7    8    5 (1,0)
            #           |  (0,0)  |
            #   (-1,-1) 0 ---4--- 1 (1,-1)
            #              (0,-1)

            # initialize elements array
            elements = np.zeros(
                (int((num_nodes_dir1 - 1) * (num_nodes_dir2 - 1) / 4), 9), dtype=int
            )

            # loop over all elements
            elem_idx = 0
            for j in range(0, num_nodes_dir2 - 2, 2):
                for i in range(0, num_nodes_dir1 - 2, 2):
                    elements[elem_idx, :] = np.array(
                        [
                            [
                                node_matrix[i, j],  # 0
                                node_matrix[i + 2, j],  # 1
                                node_matrix[i + 2, j + 2],  # 2
                                node_matrix[i, j + 2],  # 3
                                node_matrix[i + 1, j],  # 4
                                node_matrix[i + 2, j + 1],  # 5
                                node_matrix[i + 1, j + 2],  # 6
                                node_matrix[i, j + 1],  # 7
                                node_matrix[i + 1, j + 1],  # 8
                            ]
                        ]
                    )
                    elem_idx += 1

        # elements without internal nodes (quad8)
        elif element_type == "quad8":
            # local node IDs with (ξ, η):
            #
            #              (0,1)
            #   (-1, 1) 3 ---6--- 2 (1, 1)
            #           |         |
            #    (-1,0) 7         5 (1,0)
            #           |         |
            #   (-1,-1) 0 ---4--- 1 (1,-1)
            #              (0,-1)

            # initialize elements array
            elements = np.zeros(
                (int((num_nodes_dir1 - 1) * (num_nodes_dir2 - 1) / 4), 8), dtype=int
            )

            # loop over all elements
            elem_idx = 0
            for j in range(0, num_nodes_dir2 - 2, 2):
                for i in range(0, num_nodes_dir1 - 2, 2):
                    elements[elem_idx, :] = np.array(
                        [
                            [
                                node_matrix[i, j],  # 0
                                node_matrix[i + 2, j],  # 1
                                node_matrix[i + 2, j + 2],  # 2
                                node_matrix[i, j + 2],  # 3
                                node_matrix[i + 1, j],  # 4
                                node_matrix[i + 2, j + 1],  # 5
                                node_matrix[i + 1, j + 2],  # 6
                                node_matrix[i, j + 1],  # 7
                            ]
                        ]
                    )
                    elem_idx += 1

    return elements
