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

        # --------------------------------------------------------------------------------
        # Identity tensor
        I = np.eye(3)
        
        # Determinant and inverse of C
        J = np.sqrt(np.linalg.det(C))   # J = sqrt of det(C), as J = det(F) and C = F.T @ F
        C_inv = np.linalg.inv(C)
        
        # Compute the PK2 stress tensor as per equation 6.121
        S = self.μ * (I - C_inv) + self.λ * np.log(J) * C_inv


        # Compute the 4th order tensor I according to equation (6.124) (but without the redundant 1/2 and 2x)
        I_tensor = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        I_tensor[i, j, k, l] = (C_inv[i, k] * C_inv[j, l] + C_inv[i, l] * C_inv[j, k])
        
        # Compute the dyadic product
        C_inv_dyadic = np.einsum("ij,kl->ijkl", C_inv, C_inv)
        
        # Compute the constitutive stiffness tensor according to equation (6.122)
        CC = self.λ * C_inv_dyadic + (self.μ - self.λ * np.log(J)) * I_tensor
    
        # --------------------------------------------------------------------------------
        return S, CC
