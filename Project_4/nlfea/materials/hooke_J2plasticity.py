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

        ...  # TODO

        # store new history variables
        history_new_gp = np.array([ε_pl[0, 0], ε_pl[1, 1], ε_pl[0, 1], ε_pl[2, 2], α])

        return σ, DD, history_new_gp
