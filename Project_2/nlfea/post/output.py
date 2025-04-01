# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import meshio
import os


# ===================================================================================
def write_vtk(result_dir: str, increment: int, model, nodal_values):
    """
    output of vtk file with the field results, relying on the module 'meshio'
    (https://pypi.org/project/meshio/, published under the MIT licence)

    Schlömer, N. meshio: Tools for mesh files [Computer software]
    https://doi.org/10.5281/zenodo.1173115
    """

    # delete any previous vtk files in the output directory
    if increment == 0 and os.path.isdir(result_dir):
        result_dir_items = os.listdir(result_dir)
        for item in result_dir_items:
            if item.endswith(".vtk"):
                os.remove(os.path.join(result_dir, item))

    # make sure that output directory exists
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # prepare output file name
    filename = result_dir + "/nodalresults_" + f"{increment:05d}" + ".vtk"

    # translate internal element name into vtk name with lookup dictionary
    vtk_element_names = {
        "bar": "line",
        "quad4": "quad",
        "quad8": "quad8",
        "quad9": "quad9",
    }
    vtk_element_name = vtk_element_names[model.parent_element.element_type]

    # prepare elements for output
    elements = [(vtk_element_name, model.connectivity)]

    # output the domain_ids
    element_values = {"domain_id": [model.domain_ids.astype(float)]}

    # prepare nodes for output (append 3rd dimension in case of 2D model to prevent warning of meshio)
    nodes = model.nodes
    if model.dimension == 2:
        nodes = np.append(nodes, np.zeros((model.nodes.shape[0], 1)), axis=1)

    # reshape u and Fint as 2D array
    nodal_values["u"] = nodal_values["u"].reshape((-1, model.dimension))
    nodal_values["Fint"] = nodal_values["Fint"].reshape((-1, model.dimension))

    # loop over results dictionary and ensure correct scalar/ tensor/ vector output
    for label, nodes_value in nodal_values.items():
        if len(nodes_value.shape) > 1:
            if nodes_value.shape[1] == 2:
                # append 3rd dimension in case of 2D model
                nodal_values[label] = np.append(
                    nodes_value, np.zeros((model.nodes.shape[0], 1)), axis=1
                )

    # write mesh
    meshio.Mesh(nodes, elements, nodal_values, element_values).write(filename)
