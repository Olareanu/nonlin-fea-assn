# Copyright (C) 2023-2025 J. Heinzmann, O. A. Boolakee, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import math


# ===================================================================================
class Bar:
    """general 1D/ 2D/ 3D bar element definition"""

    def __init__(self, kinematics, A=1):
        # element type for ParaView output
        self.element_type = "bar"

        # cross-sectional area of element
        self.A = A

        # assign function to compute contribution to Fint and Ktan depending on kinematics
        if kinematics == "linear":
            self.Fint_Ktan = self.Fint_Ktan_linearkinematics
        elif kinematics == "nonlinear":
            self.Fint_Ktan = self.Fint_Ktan_nonlinearkinematics

    def Fext_bodyload(self, element, body_load):
        raise NotImplementedError(
            "body load not implemented for bar elements, directly use point loads instead"
        )

    def Fext_traction(self, element, edge_nodes, traction_load, direction):
        raise ValueError(
            "traction load cannot be applied for bar elements, directly use point loads instead"
        )

    # ===================================================================================
    def Fint_Ktan_linearkinematics(self, element, material, u_elem):
        # element length
        L = np.linalg.norm(element.coordinates[1, :] - element.coordinates[0, :], 2)

        # obtain rotation matrix for (1D/) 2D/ 3D
        # see lecture notes for Computational Mechanics I: Intro to FEA, pp. 42f
        # (https://ethz.ch/content/dam/ethz/special-interest/mavt/mechanical-systems/mm-dam/documents/Notes/IntroToFEA_red.pdf)
        if element.coordinates.shape[1] == 1:
            R = np.array([[1, 0], [0, 1]])
        if element.coordinates.shape[1] == 2:
            φ_e = math.atan(
                (element.coordinates[1, 1] - element.coordinates[0, 1])
                / (element.coordinates[1, 0] - element.coordinates[0, 0])
            )
            cos_φ_e = math.cos(φ_e)
            sin_φ_e = math.sin(φ_e)
            R = np.array([[cos_φ_e, sin_φ_e, 0, 0], [0, 0, cos_φ_e, sin_φ_e]])
        elif element.coordinates.shape[1] == 3:
            x_e = element.coordinates[1, :] - element.coordinates[0, :]
            e_1 = np.array([1, 0, 0])
            e_2 = np.array([0, 1, 0])
            e_3 = np.array([0, 0, 1])
            cos_φ_e_x = np.dot(x_e, e_1) / L
            cos_φ_e_y = np.dot(x_e, e_2) / L
            cos_φ_e_z = np.dot(x_e, e_3) / L
            R = np.array(
                [
                    [cos_φ_e_x, cos_φ_e_y, cos_φ_e_z, 0, 0, 0],
                    [0, 0, 0, cos_φ_e_x, cos_φ_e_y, cos_φ_e_z],
                ]
            )

        # compute contribution of element to stiffness matrix in local coordinates
        K_loc = material.E * self.A / L * np.array([[1, -1], [-1, 1]])

        # transform into global coordinates
        Ktan_elem = R.T @ K_loc @ R

        # compute contribution of element to internal force vector
        Fint_elem = Ktan_elem @ u_elem

        return Fint_elem, Ktan_elem

    # ===================================================================================
    def Fint_Ktan_nonlinearkinematics(self, element, material, u_elem):
        # position vectors of nodes in current configuration
        x = element.coordinates + np.reshape(u_elem, (2, -1))

        # necessary identities
        I = np.identity(element.coordinates.shape[1])  # noqa: E741

        # hints: the Young's modulus is stored in material.E, the cross-sectional area in self.A

        # internal force vector and stiffness matrix contribution
        Fint_elem = ...  # TODO

        Ktan_elem = ...  # TODO

        return Fint_elem, Ktan_elem
