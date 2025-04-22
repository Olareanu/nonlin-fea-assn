# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import math

from . import numerical_integration
from . import shape_functions


# ===================================================================================
class Quad:
    """general 2D quad element definition"""

    def __init__(self, kinematics, element_type, integration, T=1):
        # element type
        self.element_type = element_type

        # define thickness of element
        self.T = T

        # assign function to compute contribution to Fint and Ktan depending on kinematics
        if kinematics == "linear":
            self.Fint_Ktan = self.Fint_Ktan_linearkinematics
            self.gp_result_allocation = {
                "ψ": [0],
                "σ": [1, 2, 3, 4],
                "ε": [5, 6, 7],
            }
        elif kinematics == "nonlinear":
            self.Fint_Ktan = self.Fint_Ktan_nonlinearkinematics
            self.gp_result_allocation = {
                "Ψ": [0],
                "σ": [1, 2, 3, 4],
                "e": [5, 6, 7],
            }

        # prepare Gauss points of elements
        self.integration = integration
        self.gauss_points = numerical_integration.prepare(element_type, integration)

    # ===================================================================================
    def Fext_bodyload(
        self,
        element,
        body_load,
    ):
        # initialize internal force vector and stiffness matrix for the element
        Fext_elem = np.zeros(len(element.dofs))

        # loop over Gauss points
        for gauss_point in self.gauss_points:
            # get coordinates at Gauss point
            X_gp = gauss_point.N_scalar @ element.coordinates

            # compute Jacobian and its determinant (in reference configuration)
            J_T = gauss_point.dN_scalar.T @ element.coordinates
            detJ = np.linalg.det(J_T)

            # compute body force at the Gauss point
            b_V = body_load(X_gp[0], X_gp[1])

            # compute external force vector contribution
            Fext_elem_gp = gauss_point.N_vector @ b_V

            # numerically integrate external force vector
            Fext_elem += Fext_elem_gp * gauss_point.w * self.T * detJ

        return Fext_elem

    # ===================================================================================
    def Fext_traction(
        self,
        element,
        edge_nodes,
        traction_load,
        direction,
    ):
        # initialize internal force vector and stiffness matrix for the element
        Fext_elem = np.zeros(len(element.dofs))

        # determine edge of BC (bottom, right, top, left)
        if np.array_equal(edge_nodes[0:4], np.array([True, True, False, False])):
            edge = "bottom"
        elif np.array_equal(edge_nodes[0:4], np.array([False, True, True, False])):
            edge = "right"
        elif np.array_equal(edge_nodes[0:4], np.array([False, False, True, True])):
            edge = "top"
        elif np.array_equal(edge_nodes[0:4], np.array([True, False, False, True])):
            edge = "left"
        else:
            raise ValueError("given nodes for the edge of an element suspicious")

        # prepare numerical integration along edge
        (quadrature_coordinates, quadrature_weights) = (
            numerical_integration.gaussian_quadrature(
                dimension=1, num_gp_per_dir=math.sqrt(len(self.gauss_points))
            )
        )
        if edge == "bottom":
            ξ = quadrature_coordinates
            η = -np.ones(quadrature_coordinates.shape)
            tan_natural = np.array([1, 0])
        elif edge == "right":
            ξ = np.ones(quadrature_coordinates.shape)
            η = quadrature_coordinates
            tan_natural = np.array([0, 1])
        elif edge == "top":
            ξ = quadrature_coordinates
            η = np.ones(quadrature_coordinates.shape)
            tan_natural = np.array([-1, 0])
        elif edge == "left":
            ξ = -np.ones(quadrature_coordinates.shape)
            η = quadrature_coordinates
            tan_natural = np.array([0, -1])

        gauss_points_edge = np.vstack((ξ, η)).T

        # loop over Gauss points
        for gp_idx in range(gauss_points_edge.shape[0]):
            # compute the values of the shape functions at the current Gauss point
            (_, N_vector, dN_scalar, _) = shape_functions.N_dN(
                self.element_type, ξ[gp_idx], η[gp_idx]
            )

            # get Jacobian at edge Gauss point
            J_elem_gp = dN_scalar.T @ element.coordinates

            # get determinant of Jacobian in edge direction
            if (edge == "bottom") or (edge == "top"):
                # det_J_edge = sqrt((dx/dξ)^2 + (dx/dη)^2)
                detJ_edge_gp = np.linalg.norm(J_elem_gp[0, :])
            elif (edge == "left") or (edge == "right"):
                # det_J_edge = sqrt((dy/dξ)^2 + (dy/dη)^2)
                detJ_edge_gp = np.linalg.norm(J_elem_gp[1, :])

            # get traction vector
            if direction == "t_X":
                # traction in x-direction
                traction_vector = np.array([traction_load, 0])

            elif direction == "t_Y":
                # traction in y-direction
                traction_vector = np.array([0, traction_load])

            elif direction == "t_N":
                # compute tangential vector in physical coordinates
                tan = J_elem_gp.T @ tan_natural

                # compute normal vector on edge with cross product of tan x e_z
                n = np.array([tan[1], -tan[0]])
                n = n / np.linalg.norm(n)

                # traction in normal direction
                traction_vector = traction_load * n
            else:
                raise ValueError(f"type {direction} for Dirichlet BCs not known")

            # compute contribution to external force vector
            Fext_elem_gp = N_vector @ traction_vector

            # numerically integrate external force vector
            Fext_elem += (
                Fext_elem_gp * quadrature_weights[gp_idx] * self.T * detJ_edge_gp
            )

        return Fext_elem

    # ===================================================================================
    def Fint_Ktan_linearkinematics(self, element, material, u_elem):
        """quad element definition in small strain setting which inherits all the properties and methods from the continuum element

        refer to lecture notes for Computational Mechanics I: Intro to FEA, pp. 110f
        (https://ethz.ch/content/dam/ethz/special-interest/mavt/mechanical-systems/mm-dam/documents/Notes/IntroToFEA_red.pdf)
        """
        # initialize internal force vector and stiffness matrix for the element
        Fint_elem = np.zeros(len(element.dofs))
        Ktan_elem = np.zeros((len(element.dofs), len(element.dofs)))

        # loop over Gauss points
        for gp_idx, gauss_point in enumerate(self.gauss_points):
            # compute Jacobian, its determinant and its inverse (in reference configuration)
            J_T = gauss_point.dN_scalar.T @ element.coordinates
            detJ = np.linalg.det(J_T)
            J_invT = np.linalg.inv(J_T)

            # obtain B-matrix for small strain setting
            B_lin = (
                np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
                @ np.block([[J_invT, np.zeros((2, 2))], [np.zeros((2, 2)), J_invT]])
                @ gauss_point.dN_vector
            )

            # compute small strain tensor and transform [ε_11, ε_22, 2ε_12].T to 3D tensor (plane strain)
            ε_hat = B_lin @ u_elem
            ε = np.array(
                [[ε_hat[0], ε_hat[2] / 2, 0], [ε_hat[2] / 2, ε_hat[1], 0], [0, 0, 0]]
            )

            # get material response and rearrange into Voigt notation for 2D case
            (σ, DD, element.history_new[gp_idx, :]) = material.stress_stiffness(
                ε, element.history_old[gp_idx, :]
            )
            σ_hat = np.array([σ[0, 0], σ[1, 1], σ[0, 1]])
            D = np.array(
                [
                    [DD[0, 0, 0, 0], DD[0, 0, 1, 1], DD[0, 0, 0, 1]],
                    [DD[1, 1, 0, 0], DD[1, 1, 1, 1], DD[1, 1, 0, 1]],
                    [DD[0, 1, 0, 0], DD[0, 1, 1, 1], DD[0, 1, 0, 1]],
                ]
            )

            # compute internal force vector contribution and numerically integrate
            Fint_elem_gp = B_lin.T @ σ_hat
            Fint_elem += Fint_elem_gp * gauss_point.w * self.T * detJ

            # compute tangent stiffness matrix contribution and numerically integrate
            Ktan_elem_gp = B_lin.T @ D @ B_lin
            Ktan_elem += Ktan_elem_gp * gauss_point.w * self.T * detJ

            # postprocessing: strain energy density
            ψ = 1 / 2 * np.einsum("kl,lk", σ, ε)

            # postprocessing: store Gauss point values for postprocessing
            element.gp_result[gp_idx, :] = np.array(
                [ψ, σ[0, 0], σ[1, 1], σ[2, 2], σ[0, 1], ε[0, 0], ε[1, 1], ε[0, 1]]
            )

        return (Fint_elem, Ktan_elem)

    # ===================================================================================
    def Fint_Ktan_nonlinearkinematics(self, element, material, u_elem):
        # initialize internal force vector and stiffness matrix for the element
        Fint_elem = np.zeros(len(element.dofs))
        Ktan_elem = np.zeros((len(element.dofs), len(element.dofs)))

        # get nodal positions of element in current configuration
        x_elem = element.coordinates.flatten() + u_elem

        # loop over Gauss points
        for gp_idx, gauss_point in enumerate(self.gauss_points):
            # compute Jacobian, its determinant and its inverse (in reference configuration)
            J_T = gauss_point.dN_scalar.T @ element.coordinates
            # --------------------------------------------------------------------------------
            detJ = np.linalg.det(J_T)
            J_invT = np.linalg.inv(J_T)
            # --------------------------------------------------------------------------------

            # compute B_tilde matrix used for ∇_X (·) = B_tilde * (·)
            B_tilde = (
                np.block([[J_invT, np.zeros((2, 2))], [np.zeros((2, 2)), J_invT]])
                @ gauss_point.dN_vector
            )

            # --------------------------------------------------------------------------------
            # Compute deformation tensor
            F = x_elem.reshape(-1, 2).T @ (J_invT @ gauss_point.dN_scalar.T).T

            # Compute B using F_matrix and B_tilde
            F_11, F_12 = F[0, 0], F[0, 1]
            F_21, F_22 = F[1, 0], F[1, 1]

            F_matrix = np.array(
                [[F_11, 0, F_21, 0], [0, F_12, 0, F_22], [F_12, F_11, F_22, F_21]]
            )

            B = F_matrix @ B_tilde

            # Convert F to 3D
            F = np.array([[F[0, 0], F[0, 1], 0], [F[1, 0], F[1, 1], 0], [0, 0, 1]])

            # Compute Right Cauchy-Green tensor
            C = F.T @ F
            # --------------------------------------------------------------------------------

            # at some point, one needs to transform to the 3D state for the material routine
            # F = np.array([[F[0], F[1], 0], [F[2], F[3], 0], [0, 0, 1]])

            # get material response and bring into Voigt notation for 2D case
            # --------------------------------------------------------------------------------
            (S, DD) = material.stress_stiffness(C)

            # Transform 2nd Piola-Kirchhoff stress to Voigt notation for plane strain
            # We need [S_11, S_22, S_12] for internal force calculation
            S_hat = np.array([S[0, 0], S[1, 1], S[0, 1]])

            # Construct S_tilde matrix
            S_tilde = np.array(
                [
                    [S[0, 0], S[0, 1], 0, 0],
                    [S[1, 0], S[1, 1], 0, 0],
                    [0, 0, S[0, 0], S[0, 1]],
                    [0, 0, S[1, 0], S[1, 1]],
                ]
            )

            # Transform material tangent to Voigt notation for 2D plane strain
            D = np.array(
                [
                    [DD[0, 0, 0, 0], DD[0, 0, 1, 1], DD[0, 0, 0, 1]],
                    [DD[1, 1, 0, 0], DD[1, 1, 1, 1], DD[1, 1, 0, 1]],
                    [DD[0, 1, 0, 0], DD[0, 1, 1, 1], DD[0, 1, 0, 1]],
                ]
            )

            # Compute tangent stiffness matrix:
            K_constitutive = B.T @ D @ B  # constitutive component
            K_initialStress = B_tilde.T @ S_tilde @ B_tilde  # initial stress component

            # compute Green-Lagrange strain tensor store values for later
            E = 0.5 * (C - np.eye(3, 3))

            # compute internal force vector contribution and numerically integrate
            Fint_elem_gp = B.T @ S_hat
            Fint_elem += Fint_elem_gp * gauss_point.w * self.T * detJ

            # compute constitutive and initial stress component of stiffness matrix and numerically integrate
            Ktan_elem_gp = K_constitutive + K_initialStress
            Ktan_elem += Ktan_elem_gp * gauss_point.w * self.T * detJ

            # --------------------------------------------------------------------------------

            # postprocessing: strain energy density
            Ψ = 1 / 2 * np.einsum("kl,lk", S, E)

            # postprocessing: push forward strain and stress tensor for interpretable results, see lecture notes (6.3.12) & (6.2.14)
            e = F.T @ E @ np.linalg.inv(F)
            σ = 1 / np.linalg.det(F) * F @ S @ F.T

            # postprocessing: store Gauss point values
            element.gp_result[gp_idx, :] = np.array(
                [Ψ, σ[0, 0], σ[1, 1], σ[2, 2], σ[0, 1], e[0, 0], e[1, 1], e[0, 1]]
            )

        return Fint_elem, Ktan_elem
