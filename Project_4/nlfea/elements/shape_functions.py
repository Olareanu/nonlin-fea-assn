# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


def N_dN(element_type: str, ξ: float, η: float):
    """returns shape functions and derivatives in natural coordinates for quad element"""

    if element_type == "quad4":
        return N_dN_quad4(ξ, η)
    elif element_type == "quad8":
        return N_dN_quad8(ξ, η)
    elif element_type == "quad9":
        return N_dN_quad9(ξ, η)
    else:
        raise NotImplementedError(
            f"shape functions for element type {element_type} not available."
        )


# ===================================================================================
def N_dN_quad4(ξ: float, η: float):
    """
    returns shape functions and derivatives in natural coordinates for quad4 element,
    local node IDs with coordinates (ξ, η):

    (-1, 1) 4 ----- 3 (1, 1)
            |       |           ^ η
            |       |           |
    (-1,-1) 1 ----- 2 (1,-1)    +--> ξ

    """

    # shape functions for scalar quantity
    N_scalar = 0.25 * np.array(
        [
            (1 - ξ) * (1 - η),  # 0
            (1 + ξ) * (1 - η),  # 1
            (1 + ξ) * (1 + η),  # 2
            (1 - ξ) * (1 + η),  # 3
        ]
    )

    # shape functions for vectorial (2D) quantity
    N_vector = np.array(
        [
            [N_scalar[0], 0],
            [0, N_scalar[0]],
            [N_scalar[1], 0],
            [0, N_scalar[1]],
            [N_scalar[2], 0],
            [0, N_scalar[2]],
            [N_scalar[3], 0],
            [0, N_scalar[3]],
        ]
    )

    # derivatives of shape functions for scalar quantity
    dN_scalar = 0.25 * np.array(
        [
            [-(1 - η), -(1 - ξ)],  # 0
            [1 - η, -(1 + ξ)],  # 1
            [1 + η, 1 + ξ],  # 2
            [-(1 + η), 1 - ξ],  # 3
        ]
    )

    # derivatives of shape functions for vectorial (2D) quantity
    dN_vector = np.array(
        [
            [
                dN_scalar[0, 0],
                0,
                dN_scalar[1, 0],
                0,
                dN_scalar[2, 0],
                0,
                dN_scalar[3, 0],
                0,
            ],
            [
                dN_scalar[0, 1],
                0,
                dN_scalar[1, 1],
                0,
                dN_scalar[2, 1],
                0,
                dN_scalar[3, 1],
                0,
            ],
            [
                0,
                dN_scalar[0, 0],
                0,
                dN_scalar[1, 0],
                0,
                dN_scalar[2, 0],
                0,
                dN_scalar[3, 0],
            ],
            [
                0,
                dN_scalar[0, 1],
                0,
                dN_scalar[1, 1],
                0,
                dN_scalar[2, 1],
                0,
                dN_scalar[3, 1],
            ],
        ]
    )

    return (N_scalar, N_vector, dN_scalar, dN_vector)


# ===================================================================================
def N_dN_quad8(ξ: float, η: float):
    """
    returns shape functions and derivatives in natural coordinates for serendipity quad8 element,
    local node IDs with coordinates (ξ, η):

               (0,1)
    (-1, 1) 3 ---6--- 2 (1, 1)
            |         |
     (-1,0) 7         5 (1,0)     ^ η
            |         |           |
    (-1,-1) 0 ---4--- 1 (1,-1)    +--> ξ
            (0,-1)

    """

    # shape functions for scalar quantity
    N_scalar = np.array(
        [
            -0.25 * (1 - ξ) * (1 - η) * (1 + ξ + η),  # 0
            -0.25 * (1 + ξ) * (1 - η) * (1 - ξ + η),  # 1
            -0.25 * (1 + ξ) * (1 + η) * (1 - ξ - η),  # 2
            -0.25 * (1 - ξ) * (1 + η) * (1 + ξ - η),  # 3
            0.5 * (1 - ξ) * (1 + ξ) * (1 - η),  # 4
            0.5 * (1 + ξ) * (1 + η) * (1 - η),  # 5
            0.5 * (1 - ξ) * (1 + ξ) * (1 + η),  # 6
            0.5 * (1 - ξ) * (1 + η) * (1 - η),  # 7
        ]
    )

    # shape functions for vectorial (2D) quantity
    N_vector = np.array(
        [
            [N_scalar[0], 0],
            [0, N_scalar[0]],
            [N_scalar[1], 0],
            [0, N_scalar[1]],
            [N_scalar[2], 0],
            [0, N_scalar[2]],
            [N_scalar[3], 0],
            [0, N_scalar[3]],
            [N_scalar[4], 0],
            [0, N_scalar[4]],
            [N_scalar[5], 0],
            [0, N_scalar[5]],
            [N_scalar[6], 0],
            [0, N_scalar[6]],
            [N_scalar[7], 0],
            [0, N_scalar[7]],
        ]
    )

    # derivatives of shape functions for scalar quantity
    dN_scalar = np.array(
        [
            [-0.25 * (-1 + η) * (2 * ξ + η), -0.25 * (-1 + ξ) * (ξ + 2 * η)],  # 0
            [0.25 * (-1 + η) * (η - 2 * ξ), 0.25 * (1 + ξ) * (2 * η - ξ)],  # 1
            [0.25 * (1 + η) * (2 * ξ + η), 0.25 * (1 + ξ) * (ξ + 2 * η)],  # 2
            [-0.25 * (1 + η) * (η - 2 * ξ), -0.25 * (-1 + ξ) * (2 * η - ξ)],  # 3
            [ξ * (-1 + η), 0.5 * (1 + ξ) * (-1 + ξ)],  # 4
            [-0.5 * (1 + η) * (-1 + η), -η * (1 + ξ)],  # 5
            [-ξ * (1 + η), -0.5 * (1 + ξ) * (-1 + ξ)],  # 6
            [0.5 * (1 + η) * (-1 + η), η * (-1 + ξ)],  # 7
        ]
    )

    # derivatives of shape functions for vectorial (2D) quantity
    # fmt: off
    dN_vector = np.array(
        [
            [
                dN_scalar[0, 0], 0,
                dN_scalar[1, 0], 0,
                dN_scalar[2, 0], 0,
                dN_scalar[3, 0], 0,
                dN_scalar[4, 0], 0,
                dN_scalar[5, 0], 0,
                dN_scalar[6, 0], 0,
                dN_scalar[7, 0], 0,
            ],
            [
                dN_scalar[0, 1], 0,
                dN_scalar[1, 1], 0,
                dN_scalar[2, 1], 0,
                dN_scalar[3, 1], 0,
                dN_scalar[4, 1], 0,
                dN_scalar[5, 1], 0,
                dN_scalar[6, 1], 0,
                dN_scalar[7, 1], 0,
            ],
            [
                0, dN_scalar[0, 0],
                0, dN_scalar[1, 0],
                0, dN_scalar[2, 0],
                0, dN_scalar[3, 0],
                0, dN_scalar[4, 0],
                0, dN_scalar[5, 0],
                0, dN_scalar[6, 0],
                0, dN_scalar[7, 0],
            ],
            [
                0, dN_scalar[0, 1],
                0, dN_scalar[1, 1],
                0, dN_scalar[2, 1],
                0, dN_scalar[3, 1],
                0, dN_scalar[4, 1],
                0, dN_scalar[5, 1],
                0, dN_scalar[6, 1],
                0, dN_scalar[7, 1],
            ],
        ]
    )
    # fmt: on

    return (N_scalar, N_vector, dN_scalar, dN_vector)


# ===================================================================================
def N_dN_quad9(ξ: float, η: float):
    """
     returns shape functions and derivatives in natural coordinates for quad9 element,
     local node IDs with coordinates (ξ, η):


                   (0,1)
    (-1, 1) 3 ---6--- 2 (1, 1)
            |         |
     (-1,0) 7    8    5 (1,0)     ^ η
            |  (0,0)  |           |
    (-1,-1) 0 ---4--- 1 (1,-1)    +--> ξ
               (0,-1)

    """

    # shape functions for scalar quantity
    N_scalar = np.array(
        [
            0.5 * ξ * (ξ - 1) * 0.5 * η * (η - 1),  # 0
            0.5 * ξ * (ξ + 1) * 0.5 * η * (η - 1),  # 1
            0.5 * ξ * (ξ + 1) * 0.5 * η * (η + 1),  # 2
            0.5 * ξ * (ξ - 1) * 0.5 * η * (η + 1),  # 3
            (ξ + 1) * (1 - ξ) * 0.5 * η * (η - 1),  # 4
            0.5 * ξ * (ξ + 1) * (η + 1) * (1 - η),  # 5
            (ξ + 1) * (1 - ξ) * 0.5 * η * (η + 1),  # 6
            0.5 * ξ * (ξ - 1) * (η + 1) * (1 - η),  # 7
            (ξ + 1) * (1 - ξ) * (η + 1) * (1 - η),  # 8
        ]
    )

    # shape functions for vectorial (2D) quantity
    N_vector = np.array(
        [
            [N_scalar[0], 0],
            [0, N_scalar[0]],
            [N_scalar[1], 0],
            [0, N_scalar[1]],
            [N_scalar[2], 0],
            [0, N_scalar[2]],
            [N_scalar[3], 0],
            [0, N_scalar[3]],
            [N_scalar[4], 0],
            [0, N_scalar[4]],
            [N_scalar[5], 0],
            [0, N_scalar[5]],
            [N_scalar[6], 0],
            [0, N_scalar[6]],
            [N_scalar[7], 0],
            [0, N_scalar[7]],
            [N_scalar[8], 0],
            [0, N_scalar[8]],
        ]
    )

    # derivatives of shape functions for scalar quantity
    dN_scalar = np.array(
        [
            [(ξ - 0.5) * 0.5 * η * (η - 1), 0.5 * ξ * (ξ - 1) * (η - 0.5)],  # 0
            [(ξ + 0.5) * 0.5 * η * (η - 1), 0.5 * ξ * (ξ + 1) * (η - 0.5)],  # 1
            [(ξ + 0.5) * 0.5 * η * (η + 1), 0.5 * ξ * (ξ + 1) * (η + 0.5)],  # 2
            [(ξ - 0.5) * 0.5 * η * (η + 1), 0.5 * ξ * (ξ - 1) * (η + 0.5)],  # 3
            [(-2 * ξ) * 0.5 * η * (η - 1), (ξ + 1) * (1 - ξ) * (η - 0.5)],  # 4
            [(ξ + 0.5) * (η + 1) * (1 - η), 0.5 * ξ * (ξ + 1) * (-2 * η)],  # 5
            [(-2 * ξ) * 0.5 * η * (η + 1), (ξ + 1) * (1 - ξ) * (η + 0.5)],  # 6
            [(ξ - 0.5) * (η + 1) * (1 - η), 0.5 * ξ * (ξ - 1) * (-2 * η)],  # 7
            [(-2 * ξ) * (η + 1) * (1 - η), (ξ + 1) * (1 - ξ) * (-2 * η)],  # 8
        ]
    )

    # derivatives of shape functions for vectorial (2D) quantity
    # fmt: off
    dN_vector = np.array(
        [
            [
                dN_scalar[0, 0], 0,
                dN_scalar[1, 0], 0,
                dN_scalar[2, 0], 0,
                dN_scalar[3, 0], 0,
                dN_scalar[4, 0], 0,
                dN_scalar[5, 0], 0,
                dN_scalar[6, 0], 0,
                dN_scalar[7, 0], 0,
                dN_scalar[8, 0], 0,
            ],
            [
                dN_scalar[0, 1], 0,
                dN_scalar[1, 1], 0,
                dN_scalar[2, 1], 0,
                dN_scalar[3, 1], 0,
                dN_scalar[4, 1], 0,
                dN_scalar[5, 1], 0,
                dN_scalar[6, 1], 0,
                dN_scalar[7, 1], 0,
                dN_scalar[8, 1], 0,
            ],
            [
                0, dN_scalar[0, 0],
                0, dN_scalar[1, 0],
                0, dN_scalar[2, 0],
                0, dN_scalar[3, 0],
                0, dN_scalar[4, 0],
                0, dN_scalar[5, 0],
                0, dN_scalar[6, 0],
                0, dN_scalar[7, 0],
                0, dN_scalar[8, 0],
            ],
            [
                0, dN_scalar[0, 1],
                0, dN_scalar[1, 1],
                0, dN_scalar[2, 1],
                0, dN_scalar[3, 1],
                0, dN_scalar[4, 1],
                0, dN_scalar[5, 1],
                0, dN_scalar[6, 1],
                0, dN_scalar[7, 1],
                0, dN_scalar[8, 1],
            ],
        ]
    )
    # fmt: on

    return (N_scalar, N_vector, dN_scalar, dN_vector)
