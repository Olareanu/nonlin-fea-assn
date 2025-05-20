# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import math
from collections import namedtuple

from .shape_functions import N_dN


# ===================================================================================
def prepare(element_type, integration):
    """returns tuple of the Gauss points of an element, each with all required information"""

    # get the number of Gauss points per direction depending on the element type and chosen integration
    if element_type == "quad4":
        if integration == "full":
            num_gp_per_dir = 2
        elif integration == "reduced":
            num_gp_per_dir = 1
    elif (element_type == "quad8") or (element_type == "quad9"):
        if integration == "full":
            num_gp_per_dir = 3
        elif integration == "reduced":
            num_gp_per_dir = 2

    # get the quadrature point coordinates and weights
    (quadrature_coordinates, quadrature_weights) = gaussian_quadrature(
        2, num_gp_per_dir
    )

    # create general namedtuple as container for all the individual Gauss points
    Gauss_point = namedtuple(
        "Gauss_point",
        ["ξ", "η", "w", "N_scalar", "N_vector", "dN_scalar", "dN_vector"],
    )

    # loop over all of the quadrature points
    gauss_points = ()
    for gp_idx in range(quadrature_weights.shape[0]):
        # get parent coordinates and weight of current Gauss point
        ξ = quadrature_coordinates[gp_idx, 0]
        η = quadrature_coordinates[gp_idx, 1]
        w = quadrature_weights[gp_idx]

        # precompute the values of the shape functions at the current Gauss point
        (N_scalar, N_vector, dN_scalar, dN_vector) = N_dN(element_type, ξ, η)

        # instantiate new gauss point object with all information
        gauss_point = Gauss_point(ξ, η, w, N_scalar, N_vector, dN_scalar, dN_vector)
        gauss_points += (gauss_point,)

    return gauss_points


# ===================================================================================
def gaussian_quadrature(dimension: int, num_gp_per_dir: int):
    """returns points and weights for the selected dimension and number of Gauss points per direction"""

    # integration of 1D-domain (e.g. edge along continuum element)
    if dimension == 1:
        if num_gp_per_dir == 1:
            quadrature_coordinates = np.array([0])
            quadrature_weights = np.array([2])

        elif num_gp_per_dir == 2:
            quadrature_coordinates = np.array([-1 / math.sqrt(3), 1 / math.sqrt(3)])
            quadrature_weights = np.array([1, 1])

        elif num_gp_per_dir == 3:
            quadrature_coordinates = np.array([-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)])
            quadrature_weights = np.array([5 / 9, 8 / 9, 5 / 9])

    # integration of 2D-domain (e.g. area within continuum element)
    elif dimension == 2:
        if num_gp_per_dir == 1:
            quadrature_coordinates = np.array([[0, 0]])
            quadrature_weights = np.array([4])

        elif num_gp_per_dir == 2:
            quadrature_coordinates = np.array(
                [
                    [-1 / math.sqrt(3), -1 / math.sqrt(3)],
                    [-1 / math.sqrt(3), 1 / math.sqrt(3)],
                    [1 / math.sqrt(3), -1 / math.sqrt(3)],
                    [1 / math.sqrt(3), 1 / math.sqrt(3)],
                ]
            )
            quadrature_weights = np.array([1, 1, 1, 1])

        elif num_gp_per_dir == 3:
            quadrature_coordinates = np.array(
                [
                    [-math.sqrt(3 / 5), -math.sqrt(3 / 5)],
                    [-math.sqrt(3 / 5), 0],
                    [-math.sqrt(3 / 5), math.sqrt(3 / 5)],
                    [0, -math.sqrt(3 / 5)],
                    [0, 0],
                    [0, math.sqrt(3 / 5)],
                    [math.sqrt(3 / 5), -math.sqrt(3 / 5)],
                    [math.sqrt(3 / 5), 0],
                    [math.sqrt(3 / 5), math.sqrt(3 / 5)],
                ]
            )
            quadrature_weights = np.array(
                [
                    25 / 81,
                    40 / 81,
                    25 / 81,
                    40 / 81,
                    64 / 81,
                    40 / 81,
                    25 / 81,
                    40 / 81,
                    25 / 81,
                ]
            )

    else:
        raise ValueError(f"Quadrature for dimension {dimension} not available")

    return quadrature_coordinates, quadrature_weights
