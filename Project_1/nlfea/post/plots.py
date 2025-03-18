# Copyright (C) 2023-2025 J. Heinzmann, ETH Zurich
#
# This file is part of the `nlfea` package for the course Computational Mechanics II: Nonlinear FEA at ETH Zurich.
# This software is exclusively intended for personal, educational purposes only in the context of the above course.
# It must not be used outside of this scope without the explicit consent of the author(s).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ===================================================================================
def equilibrium_path(monitoring: dict, analytical_solution=None):
    """generates a plot for the equilibrium path"""

    # extract quantities
    u = np.abs(monitoring["u"])
    λ = monitoring["λ"]
    critical = monitoring["critical"]
    stability = monitoring["stability"]

    # generate figure axis
    fig = plt.figure()
    ax = fig.add_subplot()

    # fill unstable regions
    if not all(stability):
        # define region to be filled
        y_fill = [min(λ), max(λ)]

        # loop over unstable increments
        unstable_increments = [i for i, value in enumerate(stability) if value is False]

        for unstable_increment in unstable_increments:
            if unstable_increment == 0:
                u_start = u[unstable_increment]
            else:
                u_start = (
                    u[unstable_increment]
                    - (u[unstable_increment] - u[unstable_increment - 1]) / 2
                )

            if unstable_increment == unstable_increments[-1]:
                u_end = u[unstable_increment]
            else:
                u_end = (
                    u[unstable_increment]
                    + (u[unstable_increment + 1] - u[unstable_increment]) / 2
                )
            ax.fill_betweenx(y_fill, [u_start, u_end], color="#E2AEAB")

    # plot data
    plt.plot(u, λ, "o", label="numerical solution")

    # plot analytical solution (if existent)
    if analytical_solution is not None:
        u_plot = np.linspace(min(u), max(u), 500)
        plt.plot(u_plot, analytical_solution(u_plot), "-", label="analytical solution")

        ax.legend()

    # plot critical points (if existent)
    if not all(critical):
        critical_increments_u = [
            u[i] for i, value in enumerate(critical) if value is True
        ]
        critical_increments_λ = [
            λ[i] for i, value in enumerate(critical) if value is True
        ]

        plt.plot(critical_increments_u, critical_increments_λ, "r*")

    # labeling
    plt.title("equilibrium path")
    plt.xlabel("|u| at monitoring DOF / mm")
    plt.ylabel("λ / -")

    # plot settings
    plt.grid()

    # show plot
    plt.show()


# ===================================================================================
def force_displacement(monitoring: dict, analytical_solution=None):
    """generates a plot for the equilibrium path"""

    # extract quantities
    u = monitoring["u"]
    F = monitoring["F"]

    # generate figure axis
    fig = plt.figure()
    ax = fig.add_subplot()

    # plot data
    plt.plot(u, F, "o")

    # plot analytical solution (if existent)
    if analytical_solution is not None:
        u_plot = np.linspace(min(u), max(u), 500)
        plt.plot(u_plot, analytical_solution(u_plot), "-", label="analytical solution")

        ax.legend()

    # labeling
    plt.title("force-displacement")
    plt.xlabel("u at monitoring DOF / mm")
    plt.ylabel("F at monitoring DOF / N")

    # plot settings
    plt.grid()

    # show plot
    plt.show()


# ===================================================================================
def residuals(monitoring: dict):
    """returns a plot for the residuals"""

    # extract quantities
    res_norms = monitoring["residuals"]

    # generate figure axis
    fig = plt.figure()
    ax = fig.add_subplot()

    # loop over residual-lists of increments
    increment = 0
    for res_norms_increment in res_norms:
        iterations = range(len(res_norms_increment))
        plt.plot(iterations, res_norms_increment, label=f"increment {increment + 1:d}")
        increment += 1

    # draw a line for the solver tolerance_NRerance
    # ax.axhline(y=tolerance_NR, xmin=0, xmax=1)

    # only show legend if legend would contain less than 20 entries
    if increment <= 20:
        ax.legend()

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("residuals")
    plt.xlabel("iterations / -")
    plt.ylabel("||R||_2 / -")

    # plot settings
    plt.grid()
    plt.yscale("log")

    # show plot
    plt.show()


# ===================================================================================
def detK_displacement(monitoring: dict):
    """generates a plot for the determinant of K"""

    # extract quantities (get rid of initial state of u since detK is not known)
    u = monitoring["u"][1:]
    detK = monitoring["detK"]
    critical = monitoring["critical"]

    # generate figure axis
    plt.figure()

    # plot data
    plt.plot(u, detK, "o")

    # plot critical points (if existent)
    if not all(critical):
        critical_increments_u = [
            u[i] for i, value in enumerate(critical) if value is True
        ]
        critical_increments_detK = [
            detK[i] for i, value in enumerate(critical) if value is True
        ]

        plt.plot(critical_increments_u, critical_increments_detK, "r*")

    # labeling
    plt.title("determinant of stiffness matrix")
    plt.xlabel("u at monitoring DOF / mm")
    plt.ylabel("det(K) / -")

    # plot settings
    plt.grid()

    # show plot
    plt.show()
