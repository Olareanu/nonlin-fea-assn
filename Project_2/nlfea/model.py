# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
from collections import namedtuple


# ===================================================================================
class Model:
    """simple container for all model definitions needed throughout the finite element analysis"""

    def __init__(self):
        # finite element
        self.parent_element = None

        # mesh
        self.dimension = None
        self.nodes = None
        self.elements = ()
        self.connectivity = None
        self.domain_ids = None

        # constitutive law
        self.material = None

        # degrees of freedom
        self.dofs = None
        self.num_dofs = None

        # boundary conditions
        self.dirichlet_bcs = None
        self.neumann_bcs = None
        self.body_loads = None
        self.point_loads = None
        self.spring_bcs = None
        self.contact_bcs = None

        # precomputed quantities to optimize runtime
        self.Ktan_rows = None
        self.Ktan_cols = None
        self.M = None

    # ===================================================================================
    def prepare(self):
        """prepares all necessary definitions for the model"""

        # get dimension of model (1D/ 2D/ 3D) based on the number of columns in the nodes array
        self.dimension = self.nodes.shape[1]

        # assign default domain ID if it is not explicitly set by the user
        if self.domain_ids is None:
            self.domain_ids = np.zeros(self.connectivity.shape[0], dtype=int)

        # make tuple out of the materials
        if type(self.material) is not tuple:
            num_domains = max(self.domain_ids) + 1
            self.material = (self.material,) * num_domains

        # total degrees of freedom (number of nodes * number of coordinates per node)
        self.num_dofs = self.nodes.shape[0] * self.nodes.shape[1]

        # prepare row and column indices for the fast creation of sparse matrix
        # creation of the sparse matrix in IJV-format, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        # each element has its unchanging position in Ktan, hence its corresponding rows and cols must only be determined once
        num_elements = self.connectivity.shape[0]
        num_dofs_per_element = self.connectivity.shape[1] * self.dimension
        num_entries = num_elements * num_dofs_per_element**2
        self.Ktan_rows = np.zeros(num_entries)
        self.Ktan_cols = np.zeros(num_entries)

        # loop over the elements of the mesh and create individual instances of the following namedtuple
        Element = namedtuple(
            "Element",
            [
                "nodes",
                "coordinates",
                "domain_id",
                "dofs",
                "Ktan_sparse_idx",
                "gp_result",
            ],
        )
        for element_idx, element_nodes in enumerate(self.connectivity):
            # extract the coordinates of the element
            element_coordinates = self.nodes[element_nodes, :]

            # get the domain ID of the element
            element_domain_id = self.domain_ids[element_idx]

            # determine the DOFs of the element by looping over its nodes
            element_dofs = []
            for element_node in element_nodes:
                element_dofs.extend(
                    element_node * self.dimension + range(self.dimension)
                )

            # compute indices for assembly of sparse Ktan
            Ktan_sparse_start = element_idx * num_dofs_per_element**2
            Ktan_sparse_end = (element_idx + 1) * num_dofs_per_element**2
            Ktan_sparse_idx = np.arange(Ktan_sparse_start, Ktan_sparse_end)
            self.Ktan_rows[Ktan_sparse_idx] = np.tile(
                element_dofs, (num_dofs_per_element, 1)
            ).T.flatten()
            self.Ktan_cols[Ktan_sparse_idx] = np.tile(
                element_dofs, (num_dofs_per_element, 1)
            ).flatten()

            # prepare the array storing the results at the Gauss points based on the gp_result_allocation property
            if hasattr(self.parent_element, "gp_result_allocation"):
                gp_result_allocation = self.parent_element.gp_result_allocation
                num_gp_result = (
                    max(
                        [max(gp_result_allocation[idx]) for idx in gp_result_allocation]
                    )
                    + 1
                )
                gp_result_array = np.zeros(
                    (len(self.parent_element.gauss_points), num_gp_result)
                )
            else:
                gp_result_array = None

            # instantiate new element object with current nodes and DOFs and append to elements
            element = Element(
                nodes=element_nodes,
                coordinates=element_coordinates,
                domain_id=element_domain_id,
                dofs=element_dofs,
                Ktan_sparse_idx=Ktan_sparse_idx,
                gp_result=gp_result_array,
            )
            self.elements += (element,)

        # print information
        print(f"model dimension:\t\t{self.dimension}")
        print(f"chosen element:\t\t\t{self.parent_element.element_type}")
        print(f"number of nodes:\t\t{self.nodes.shape[0]}")
        print(f"number of elements:\t\t{len(self.elements)}")
        print(f"number of DOFs:\t\t\t{self.num_dofs}\n")
