# Copyright (C) 2023-2025 J. Heinzmann, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import math


# ===================================================================================
# ===================================================================================
# ===================================================================================
def pressurized_cylinder(material, r_outer, r_inner, t_bar, u):
    """
    returns closed form solution for perfect plasticity in pressurized cylinder, based on:

    EA de Souza Neto (2008): Computational Methods for Plasticity, pp. 245ff, [based on Hill, 1950]
    """

    t = np.zeros_like(u)
    Y = material.σ_0 / math.sqrt(3)
    c = np.sqrt(u * material.E * r_outer / (2 * Y * (1 - material.ν**2)))
    elastic_flag = c < r_inner
    t[elastic_flag] = (
        u[elastic_flag]
        * material.E
        * ((r_outer / r_inner) ** 2 - 1)
        / (2 * r_outer * (1 - material.ν**2))
    )
    plastic_flag = (c >= r_inner) & (c < r_outer)
    t[plastic_flag] = Y * (
        2 * np.log(c[plastic_flag] / r_inner) + (1 - (c[plastic_flag] / r_outer) ** 2)
    )
    t[~(elastic_flag | plastic_flag)] = 2 * Y * math.log(r_outer / r_inner)

    t = t / abs(t_bar)

    return t
