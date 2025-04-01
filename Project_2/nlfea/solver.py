# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np


# ===================================================================================
class Solver:
    """simple container for all settings and parameters needed for the finite element solver"""

    def __init__(self):
        # incremental solver
        self.num_increments = 10

        # Newton-Raphson scheme
        self.tolerance_nr = 1e-6
        self.max_iter = 10

        # control method (either load, displacement, arclength)
        self.control_method = None
        self.control_size = 0.1

        # stability
        self.stability_analysis = False
        self.tolerance_detK = 1e-5
        self.max_bisection_iter = 20
        self.perturbation = False

        # contact boundary conditions
        self.contact_penalty = 100
        self.contact_tolerance_edge = 1e-3

        # postprocessing
        self.monitoring_dof = None
        self.monitoring = {
            "u": [0],
            "F": [0],
            "λ": [0],
            "detK": [None],  # merely a placeholder, we don't actually compute this
            "stability": [True],
            "critical": [False],
            "residuals": [],
        }
        self.result_dir = None
