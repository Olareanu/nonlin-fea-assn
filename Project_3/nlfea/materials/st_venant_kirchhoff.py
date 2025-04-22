# Copyright (C) 2023-2025 J. Heinzmann, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


# ===================================================================================
class StVenantKirchhoff:
    """
    St. Venant Kirchhoff material model for large strains

    strain energy density
    Ψ = 1/2 λ * tr**2(E) + μ * E:E
    """

    def __init__(self, λ: float = 121.154, μ: float = 80.769):
        # compute first and second Lamé parameter
        self.λ = λ
        self.μ = μ

    def stress_stiffness(self, C: np.array):
        """
        returns the PK2 stress tensor S and the material tangent stiffness tensor CC

        the right Cauchy-Green tensor is expected to have the format
        #     [C_11 C_12 C_13]
        # C = [C_12 C_22 C_23]
        #     [C_13 C_23 C_33]
        """

        # get identity tensors
        I = np.eye(3, 3)  # noqa: E741
        II = np.einsum("ij,kl->ijkl", I, I)
        II_sym = (
            1 / 2 * (np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I))
        )

        # compute Green-Lagrange strain tensor
        E = 0.5 * (C - np.eye(3, 3))

        # compute stress tensor
        S = self.λ * np.trace(E) * I + 2 * self.μ * E

        # constitutive matrix
        CC = self.λ * II + 2 * self.μ * II_sym

        return S, CC
