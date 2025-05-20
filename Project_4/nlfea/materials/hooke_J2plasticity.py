# Copyright (C) 2023-2025 J. Heinzmann, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import math


# ===================================================================================
class Hooke_J2plasticity:
    """
    Hooke material model with linear isotropic hardening plasticity model for small strains

    strain energy density
    ψ = 1/2 λ * tr**2(ε) + μ * ε:ε

    von Mises yield criterion (J2-plasticity)
    f = sqrt(3/2) ||σ_dev|| - (σ_0 + Kα)
    """

    def __init__(self, E: float = 210, ν: float = 0.3, σ_0: float = 250, K: float = 50):
        self.E = E
        self.ν = ν
        self.σ_0 = σ_0
        self.K = K

        # compute shear and bulk modulus
        self.μ = E / (2 * (1 + ν))
        self.κ = E / (3 * (1 - 2 * ν))

        # history variable allocation
        self.history_allocation = {
            "ε_pl": [0, 1, 2, 3],
            "α": [4],
        }

    def stress_stiffness(self, ε: np.array, history_old: np.ndarray):
        """
        returns the Cauchy stress tensor σ and the algorithmic material tangent modulus tensor DD

        the strain tensor is expected to have the format
        #     [ε_11 ε_12 ε_13]
        # ε = [ε_12 ε_22 ε_23]
        #     [ε_13 ε_23 ε_33]
        """

        # unpack history variables
        #        [ε_pl_11 ε_pl_12    0   ]
        # ε_pl = [ε_pl_12 ε_pl_22    0   ]
        #        [   0       0    ε_pl_33]
        ε_pl = np.array(
            [
                [history_old[0], history_old[2], 0],
                [history_old[2], history_old[1], 0],
                [0, 0, history_old[3]],
            ]
        )
        α = history_old[4]

        # ------------------------------------------------------------------------------
        # as per the lecture notes, Table 8.7

        # get projection tensors (like hooke.py)
        I = np.eye(3, 3)
        II = np.einsum("ij,kl->ijkl", I, I)
        II_sym = (
            1.0
            / 2.0
            * (np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I))
        )
        II_sph = 1.0 / 3.0 * np.einsum("ij,kl->ijkl", I, I)
        II_dev = II_sym - II_sph

        # compute elasticity tensor
        CC = 2 * self.μ * II_dev + 3 * self.κ * II_sph

        # compute trial quantities
        sigma_tr_new = np.einsum("ijkl,kl->ij", CC, ε - ε_pl)

        sigma_tr_dev_new = sigma_tr_new - np.trace(sigma_tr_new) * I / 3
        norm_sigma_tr_dev_new = np.linalg.norm(sigma_tr_dev_new)
        f_tr_new = np.sqrt(3 / 2) * norm_sigma_tr_dev_new - (self.σ_0 + self.K * α)

        if f_tr_new <= 0:
            # elastic step
            σ = sigma_tr_new
            DD = CC

        else:
            # plastic step
            n = sigma_tr_dev_new / norm_sigma_tr_dev_new

            # plastic quantities
            delta_gamma = f_tr_new / (3 * self.μ + self.K)
            ε_pl = ε_pl + np.sqrt(3 / 2) * delta_gamma * n
            α = α + delta_gamma

            # compute new stress tensor
            sigma_dev_new = sigma_tr_dev_new * (
                1 - np.sqrt(3 / 2) * 2 * self.μ * delta_gamma / norm_sigma_tr_dev_new
            )
            sigma_sph_new = self.κ * np.trace(ε) * I
            σ = sigma_dev_new + sigma_sph_new

            # compute algorithmic tangent modulus
            DD = (
                CC
                - 2
                * self.μ
                / (1 + self.K / (3 * self.μ))
                * np.einsum("ij,kl->ijkl", n, n)
                - delta_gamma
                * np.sqrt(3 / 2)
                * 4
                * self.μ**2
                / norm_sigma_tr_dev_new
                * (II_dev - np.einsum("ij,kl->ijkl", n, n))
            )

        # ------------------------------------------------------------------------------
        # store new history variables
        history_new_gp = np.array([ε_pl[0, 0], ε_pl[1, 1], ε_pl[0, 1], ε_pl[2, 2], α])

        return σ, DD, history_new_gp
