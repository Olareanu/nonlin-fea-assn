# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import math
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse import linalg as sparse_linalg
from scipy.interpolate import interpn

from ..elements import numerical_integration


# ===================================================================================
def project(model):
    """
    project the values to the nodal positions

    based on the L2-projection method as explained in
    E. Oñate (2009): Structural Analysis with the Finite Element Method. Linear Statics. Vol. 1: Basis and Solids
    https://doi.org/10.1007/978-1-4020-8733-2
    Section 9.8: Computation of nodal stresses, pp. 329 ff

    minimize error ϵ = θ_smoothened - θ_Gausspoints
    by solving a equation system θ_smoothened = M^(-1) * g
    with θ_smoothened being the globally smoothened values of θ, which are given at the Gausspoints from the FEM
    """

    nodal_values_projected = {}

    # project the Gauss point results to the nodes
    if hasattr(model.parent_element, "gp_result_allocation"):
        for (
            gp_result_label,
            gp_result_indices,
        ) in model.parent_element.gp_result_allocation.items():
            # initialize empty array to be filled
            gp_result_projected = np.zeros(
                (model.nodes.shape[0], len(gp_result_indices))
            )

            for i, gp_result_index in enumerate(gp_result_indices):
                # compute vector g for the L2-projection
                g_i = compute_g(model, "gp_result", gp_result_index)

                # compute L2-projection and store results
                gp_result_projected[:, i] = sparse_linalg.spsolve(model.M, g_i)

            nodal_values_projected[gp_result_label] = gp_result_projected

    # project the history variables at the Gauss point results to the nodes
    if hasattr(model.material[0], "history_allocation"):
        for history_label, history_indices in model.material[
            0
        ].history_allocation.items():
            # initialize empty array to be filled
            history_projected = np.zeros((model.nodes.shape[0], len(history_indices)))

            for i, history_index in enumerate(history_indices):
                # compute vector g for the L2-projection
                g_i = compute_g(model, "history_new", history_index)

                # compute L2-projection and store results
                history_projected[:, i] = sparse_linalg.spsolve(model.M, g_i)

            nodal_values_projected[history_label] = history_projected

    return nodal_values_projected


# ===================================================================================
def compute_M(model):
    """
    returns matrix M for the L2-projection

    always consider a fully integrated element, since otherwise the system is underdetermined
    (trying to obtain e.g. 9 nodal values from 4 Gauss point values for reduced Quad4 element)
    """

    # prepare numerical integration for fully integrated element
    gauss_points = numerical_integration.prepare(
        model.parent_element.element_type, "full"
    )

    # initialize sparse matrix M
    num_elements = model.connectivity.shape[0]
    num_dofs_per_element = model.connectivity.shape[1]
    num_entries = num_elements * num_dofs_per_element**2
    M_rows = np.zeros(num_entries)
    M_cols = np.zeros(num_entries)
    M_vals = np.zeros(num_entries)

    # loop over elements to assemble contributions
    for element_idx, element in enumerate(model.elements):
        M_elem = np.zeros((element.nodes.shape[0], element.nodes.shape[0]))

        # loop over Gauss points to integrate mass matrix contribution of element
        for gauss_point in gauss_points:
            # compute Jacobian
            J_elem_gp = gauss_point.dN_scalar.T @ element.coordinates
            detJ_elem_gp = np.linalg.det(J_elem_gp)

            M_elem_gp = np.outer(gauss_point.N_scalar, gauss_point.N_scalar)
            M_elem += M_elem_gp * gauss_point.w * detJ_elem_gp

        # assemble element contributions at corresponding DOFs
        M_sparse_start = element_idx * num_dofs_per_element**2
        M_sparse_end = (element_idx + 1) * num_dofs_per_element**2
        M_sparse_idx = np.arange(M_sparse_start, M_sparse_end)
        M_rows[M_sparse_idx] = np.tile(
            element.nodes, (num_dofs_per_element, 1)
        ).T.flatten()
        M_cols[M_sparse_idx] = np.tile(
            element.nodes, (num_dofs_per_element, 1)
        ).flatten()
        M_vals[M_sparse_idx] = M_elem.flatten()

    # convert to sparse matrix and convert to CSR format
    M = coo_array((M_vals, (M_rows, M_cols)), shape=(model.nodes.shape[0],) * 2).tocsr()

    return M


# ===================================================================================
def compute_g(model, variable_type, i):
    """
    returns vector g for the L2-projection

    always consider a fully integrated element, since otherwise the system is underdetermined
    (trying to obtain e.g. 9 nodal values from 4 Gauss point values for reduced Quad4 element)

    this necessitates a projection of the original Gauss point values of the reduced to the fully integrated element
    (done in the parent domain)
    """

    # initialize vector g for L2-projection of values
    g = np.zeros(model.nodes.shape[0])

    # check if full or reduced integration element was used for the computations
    if model.parent_element.integration == "full":
        # loop over elements to assemble contributions
        for element in model.elements:
            # loop over Gauss points of original, reduced integration element
            for gp_idx, gauss_point in enumerate(model.parent_element.gauss_points):
                # compute Jacobian and its determinant
                J_elem_gp = gauss_point.dN_scalar.T @ element.coordinates
                detJ_elem_gp = np.linalg.det(J_elem_gp)

                # integrate and assemble element contributions at corresponding nodes
                g_elem_gp = (
                    gauss_point.N_scalar * getattr(element, variable_type)[gp_idx, i]
                )
                g[element.nodes] += g_elem_gp * gauss_point.w * detJ_elem_gp

    else:
        # prepare gauss points of surrogate element with full integration for extrapolation
        gauss_points_surrogate = numerical_integration.prepare(
            model.parent_element.element_type, "full"
        )
        num_gp_per_dir_full = int(math.sqrt(len(gauss_points_surrogate)))
        num_gp_per_dir_reduced = int(math.sqrt(len(gauss_points_surrogate)) - 1)
        (gp_coordinates_surrogate, _) = numerical_integration.gaussian_quadrature(
            2, num_gp_per_dir_full
        )
        (gp_coordinates_reduced, _) = numerical_integration.gaussian_quadrature(
            2, num_gp_per_dir_reduced
        )
        gp_coordinates = (np.unique(gp_coordinates_reduced),) * 2

        # loop over elements to assemble contributions
        for element in model.elements:
            # first  project Gauss point values of the reduced to the fully integrated element
            gausspoint_values_elem = project_gps(
                gp_coordinates,
                getattr(element, variable_type)[:, i].reshape(
                    num_gp_per_dir_reduced, num_gp_per_dir_reduced
                ),
                gp_coordinates_surrogate,
            )

            # loop over Gauss points of surrogate full integration element
            for gp_idx, gauss_point_surrogate in enumerate(gauss_points_surrogate):
                # compute Jacobian and its determinant
                J_elem_gp = gauss_point_surrogate.dN_scalar.T @ element.coordinates
                detJ_elem_gp = np.linalg.det(J_elem_gp)

                # integrate and assemble element contributions at corresponding nodes
                g_elem_gp = (
                    gauss_point_surrogate.N_scalar * gausspoint_values_elem[gp_idx]
                )
                g[element.nodes] += g_elem_gp * gauss_point_surrogate.w * detJ_elem_gp

    return g


# ===================================================================================
def project_gps(old_points, old_values, new_points):
    """inter-/ extrapolates old_values given at old_points to new_points"""

    if old_points[0].shape[0] == 1:
        # single old_point to multiple new_points: "inherit" old_value for all new_values
        new_values = np.ones(new_points.shape[0]) * old_values[0, 0]

    else:
        # perform linear inter- and extrapolation from old_points to new_points
        new_values = interpn(
            old_points,
            old_values,
            new_points,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    return new_values
