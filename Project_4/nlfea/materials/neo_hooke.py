# Copyright (C) 2023-2025 J. Heinzmann, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import math


# ===================================================================================
class NeoHooke:
    """
    neo-Hooke material model for large strains (compressible)

    strain energy density
    Ψ = 1/2 μ (I^1_C - 3) - μ ln(J) + 1/2 λ (ln(J))**2
    """

    def __init__(self, λ: float = 121.154, μ: float = 80.769):
        # compute shear and bulk modulus
        self.μ = μ
        self.λ = λ

    def stress_stiffness(self, C: np.array):
        """
        returns the PK2 stress tensor S and the material tangent stiffness tensor CC

        the right Cauchy-Green tensor is expected to have the format
        #     [C_11 C_12 C_13]
        # C = [C_12 C_22 C_23]
        #     [C_13 C_23 C_33]
        """

        # get identity tensor
        I = np.eye(3, 3)  # noqa: E741

        # compute relevant quantities related to C
        I3C = np.linalg.det(C)
        J = math.sqrt(I3C)
        C_inv = np.linalg.inv(C)

        # compute the PK2 stress tensor
        S = self.μ * (I - C_inv.T) + self.λ * np.log(J) * C_inv.T

        # compute the constitutive stiffness matrix
        # fmt: off
        CC = (
            (self.μ - self.λ * np.log(J)) * ( np.einsum("lm,kn->klmn", C_inv, C_inv) + np.einsum("ln,km->klmn", C_inv, C_inv) )
            + self.λ * np.einsum("lk,nm->klmn", C_inv, C_inv)
        )
        # fmt: on

        return S, CC
