# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import pyvista as pv
import glob
import os
import numpy as np


def truss(
    vtk_dir: str,
    field: str = "u",
    deformation_magnification: float = 1.0,
):
    """
    reads vtk files for trusses in a given folder and visualizes them interactively in a new window
    """

    # get list of all vtk files in specified folder
    try:
        vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "*.vtk")))
    except FileNotFoundError:
        print("folder not found")

    # check if vtk files are found
    if not vtk_files:
        raise FileNotFoundError(f"no VTK files found in {vtk_dir}")

    # read first vtk file to determine 2D or 3D
    mesh_initial = pv.read(vtk_files[0])
    if all(np.isclose(mesh_initial.points[:, 2], 0.0)):
        dim = 2
    else:
        dim = 3

    # create plotter object
    plotter = pv.Plotter()

    # function to create mesh at given increment
    def create_mesh(increment):
        # load mesh file
        vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "*.vtk")))
        mesh = pv.read(vtk_files[int(increment)])

        # apply deformation
        if deformation_magnification != 0.0:
            displacement = mesh.point_data["u"]
            mesh.points += deformation_magnification * displacement

        # add mesh to plotter
        plotter.add_mesh(
            mesh,
            name="mesh",
            scalars=field,
            style="wireframe",
            cmap="turbo",
            line_width=5.0,
            scalar_bar_args={"title": field},
        )

    # add widget to select increment
    plotter.add_slider_widget(
        create_mesh,
        [0, len(vtk_files) - 1],
        value=len(vtk_files) - 1,
        title="increment",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9),
        color="black",
        title_opacity=0.75,
        fmt="%0.0f",
        style="modern",
        interaction_event="always",
    )

    # add text for magnification
    if deformation_magnification != 1.0:
        plotter.add_text(
            f"deformation magnification: {deformation_magnification}",
            position="upper_left",
            font_size=10,
            color="red",
        )

    # configure and show plotter
    plotter.show_axes()
    plotter.enable_parallel_projection()
    if dim == 2:
        plotter.enable_2d_style()
        plotter.set_viewup([0, 1, 0])
    plotter.show()


def continuum(
    vtk_dir: str,
    field: str = "u",
    deformation_magnification: float = 1.0,
):
    """
    reads vtk files for continua in a given folder and visualizes them interactively in a new window
    """

    # get list of all vtk files in specified folder
    try:
        vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "*.vtk")))
    except FileNotFoundError:
        print("folder not found")

    # check if vtk files are found
    if not vtk_files:
        raise FileNotFoundError(f"no VTK files found in {vtk_dir}")

    # create plotter object
    plotter = pv.Plotter()

    # function to create mesh at given increment
    def create_mesh(increment):
        # load mesh file
        vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "*.vtk")))
        mesh = pv.read(vtk_files[int(increment)])

        # apply deformation
        if deformation_magnification != 0.0:
            displacement = mesh.point_data["u"]
            mesh.points += deformation_magnification * displacement

        # # extract outer element edges to plot them separately (otherwise PyVista will plot also internal edges)
        surface = mesh.separate_cells().extract_surface(nonlinear_subdivision=4)
        edges = surface.extract_feature_edges()

        # add edges and mesh to plotter
        plotter.add_mesh(
            edges, name="edges", show_edges=True, color="black", line_width=2.0
        )
        plotter.add_mesh(
            surface,
            name="surface",
            scalars=field,
            style="surface",
            show_edges=False,
            cmap="turbo",
            scalar_bar_args={"title": field},
        )

    # add widget to select increment
    plotter.add_slider_widget(
        create_mesh,
        [0, len(vtk_files) - 1],
        value=len(vtk_files) - 1,
        title="increment",
        pointa=(0.2, 0.9),
        pointb=(0.8, 0.9),
        color="black",
        title_opacity=0.75,
        fmt="%0.0f",
        style="modern",
        interaction_event="always",
    )

    # add text for magnification
    if deformation_magnification != 1.0:
        plotter.add_text(
            f"deformation magnification: {deformation_magnification}",
            position="upper_left",
            font_size=10,
            color="red",
        )

    # configure and show plotter
    plotter.show_axes()
    plotter.enable_parallel_projection()
    plotter.enable_2d_style()
    plotter.set_viewup([0, 1, 0])
    plotter.show()
