# Copyright (C) 2023-2025 J. Heinzmann, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


# ===================================================================================
class Hooke:
    """
    Hooke material model for small strains

    strain energy density
    ψ = 1/2 λ * tr**2(ε) + μ * ε:ε
    """

    def __init__(self, E: float = 210, ν: float = 0.3):
        self.E = E
        self.ν = ν

        # compute Lamé parameters
        self.λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
        self.μ = (E) / (2 * (1 + ν))

    def stress_stiffness(self, ε: np.array, history: np.ndarray):
        """
        returns the Cauchy stress tensor σ and the material tangent stiffness tensor CC

        the small strain tensor is expected to have the format
        #     [ε_11 ε_12 ε_13]
        # ε = [ε_12 ε_22 ε_23]
        #     [ε_13 ε_23 ε_33]
        """

        # get identity tensors
        I = np.eye(3, 3)  # noqa: E741
        II = np.einsum("ij,kl->ijkl", I, I)
        II_sym = (
            1 / 2 * (np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I))
        )

        # compute stress tensor
        σ = self.λ * np.trace(ε) * I + 2 * self.μ * ε

        # constitutive matrix
        CC = self.λ * II + 2 * self.μ * II_sym

        return σ, CC, history
