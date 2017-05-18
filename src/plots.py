from utils import simulate, measure, classical_smc, ekf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math

PATH='../report/figures/'

def plot_3d_trajectory(export=False, filename='q2-3d-trajectory.pdf'):
    """Simulate the 3d-system for 50 seconds and make a 3D plot of the
    trajectory.

    Arguments:
    export -- indicate if the plot should be PDF exported and saved (default
    False)
    filename -- exported PDF filename (default q2-3d-trajectory.pdf)
    """

    xs, ys, zs = simulate(50)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(xs[0], ys[0], zs[0], color='g')
    ax.plot(xs, ys, zs)
    plt.show()

    if export:
        fig.savefig(PATH + filename)

def plot_mes_vs_real(export=False, filename='q2-mes-vs-real.pdf'):
    """Compare noisy measurements of the first coordinate of the particle
    positions with the actual simulated first coordinate.

    Arguments:
    export -- indicate if the plot should be PDF exported and saved (default
    False)
    filename -- exported PDF filename (default q2-mes-vs-real.pdf)
    """

    t_tot = 50
    dt = 0.001
    ts = 0.01
    L = int(ts/dt)
    sigma_m = 1

    a = np.arange(0, int(t_tot/dt) + 1, 1)

    xs, _, _ = simulate(t_tot)
    xs_m = measure(xs, L)

    fig, ax = plt.subplots()
    ax.plot(a, xs, 'b', label='First coordinate trajectory')
    ax.plot(a[:-1:L], xs_m, 'g.', label='Noisy measurements', markersize=4.0)
    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    plt.show()

    if export:
        fig.savefig(PATH + filename)

def plot_samples_hist(export=False, filename='q3-samples-hist.pdf'):
    t_tot = 16
    dt = 0.001
    ts = 0.01
    L = int(ts/dt)

    xs, ys, zs = simulate(t_tot, dt=dt)
    xs_m = measure(xs, L)

    x_tilde, y_tilde, z_tilde, x, y, z, wxs = classical_smc(xs_m, t_tot, L,
                                                            dt=dt, ts=ts)

    # Histograms for x
    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.hist(x[int(5/ts), :], color='b', align='mid', alpha=0.5)
    plt.axvline(xs[int(5/dt)], color='g', linestyle='dashed', linewidth=2)

    plt.subplot(1, 3, 2)
    plt.hist(x[int(10/ts), :], color='b', align='mid', alpha=0.5)
    plt.axvline(xs[int(10/dt)], color='g', linestyle='dashed', linewidth=2)

    plt.subplot(1, 3, 3)
    plt.hist(x[int(15/ts), :], color='b', align='mid', alpha=0.5)
    plt.axvline(xs[int(15/dt)], color='g', linestyle='dashed', linewidth=2)

    # Histograms for y
    plt.figure(2)
    plt.subplot(1, 3, 1)
    plt.hist(y[int(5/ts), :], color='b', align='mid', alpha=0.5)
    plt.axvline(ys[int(5/dt)], color='g', linestyle='dashed', linewidth=2)

    plt.subplot(1, 3, 2)
    plt.hist(y[int(10/ts), :], color='b', align='mid', alpha=0.5)
    plt.axvline(ys[int(10/dt)], color='g', linestyle='dashed', linewidth=2)

    plt.subplot(1, 3, 3)
    plt.hist(y[int(15/ts), :], color='b', align='mid', alpha=0.5)
    plt.axvline(ys[int(15/dt)], color='g', linestyle='dashed', linewidth=2)

    # Error function
    plt.figure(3)
    x_real = np.empty((int(t_tot/ts), 3))
    x_real[:, 0] = xs[:-1:L]
    x_real[:, 1] = ys[:-1:L]
    x_real[:, 2] = zs[:-1:L]

    a = np.arange(0, int(t_tot/dt) + 1, 1)
    err = np.linalg.norm(x_real - wxs, axis=1)
    plt.plot(a[:-1:L], err)

    # Tracking degenerancy
    plt.figure(4)
    var = np.var(x_tilde, axis=1)
    plt.plot(a[:-1:L], var)

    # Particles
    fig = plt.figure(5)
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot(x_tilde[int(5/ts), :], y_tilde[int(5/ts), :], z_tilde[int(5/ts), :],
           'b.', label='Particles before resampling', markersize=7.0)
    ax.plot(x[int(5/ts), :], y[int(5/ts), :], z[int(5/ts), :],
           'rx', label='Particles after resampling', alpha=0.5, markersize=8.0)

    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    fig = plt.figure(6)
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot(x_tilde[int(15/ts), :], y_tilde[int(15/ts), :], z_tilde[int(15/ts), :],
           'b.', label='Particles before resampling', markersize=7.0)
    ax.plot(x[int(15/ts), :], y[int(15/ts), :], z[int(15/ts), :],
           'rx', label='Particles after resampling', alpha=0.5, markersize=8.0)

    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    # Trajectory of first coordinates
    fig, ax = plt.subplots()
    ax.plot(a, xs, 'b', label='First coordinate trajectory (real)')
    ax.plot(a[:-1:L], xs_m, 'g.', label='Noisy measurements', markersize=4.0)
    ax.plot(a[:-1:L], wxs[:, 0], 'rx', label='CSMC output',
            markersize=4.0)

    """
    a = a[:-1:L]
    for i in range(len(a)):
        iax.plot(a[i]*np.ones(len(x[i, :])), x_tilde[i, :], 'c.', markersize=5.0)
        ax.plot(a[i]*np.ones(len(x[i, :])), x[i, :], 'b.', markersize=1.0)
    """

    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    plt.show()

def plot_ekf(export=False, filename='q6-samples-ekf.pdf'):
    t_tot = 16
    dt = 0.001
    ts = 0.01
    sigma_u = 10
    L = int(ts/dt)

    xs, ys, zs = simulate(t_tot)
    xs_m = measure(xs, L)

    x_tilde = np.array([X[0] for X in ekf(xs_m, t_tot, L, sigma_u=sigma_u)])
    
    a = np.arange(0, int(t_tot/dt) + 1, 1)
    fig, ax = plt.subplots()
    ax.plot(a, xs, 'b', label='First coordinate trajectory (real)')
    ax.plot(a[:-1:L], xs_m, 'g.', label='Noisy measurements', markersize=4.0)
    ax.plot(a[:-1:L], np.mean(x_tilde, axis=1), 'rx', label='EKF output',
            markersize=4.0)
    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    plt.show()


def main():
    #plot_3d_trajectory(False)
    #plot_mes_vs_real(False)
    plot_samples_hist()
    #plot_ekf()

if __name__ == "__main__":
    main()
