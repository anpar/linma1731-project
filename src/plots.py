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

    xs, ys, zs = simulate(t_tot)
    xs_m = measure(xs, L)

    x_tilde, _, _, wxs = classical_smc(xs_m, t_tot, L)
    print(x_tilde[int(5/ts), :])
    print(xs[int(5/dt)])

    result = plt.hist(x_tilde[int(5/ts), :], color='b', align='mid')
    plt.axvline(np.mean(x_tilde[int(5/ts), :]), color='k', linestyle='dashed', linewidth=2)
    plt.axvline(xs[int(5/dt)], color='g', linestyle='dashed', linewidth=2)
    plt.show()

    result = plt.hist(x_tilde[int(10/ts), :], bins=20, color='b')
    plt.axvline(np.mean(x_tilde[int(10/ts), :]), color='k', linestyle='dashed',
                                linewidth=2)
    plt.axvline(xs[int(10/dt)], color='g', linestyle='dashed', linewidth=2)
    plt.show()

    result = plt.hist(x_tilde[int(15/ts), :], bins=20, color='b')
    plt.axvline(np.mean(x_tilde[int(15/ts), :]), color='k', linestyle='dashed',
                                linewidth=2)
    plt.axvline(xs[int(15/dt)], color='g', linestyle='dashed', linewidth=2)
    plt.show()

    a = np.arange(0, int(t_tot/dt) + 1, 1)
    fig, ax = plt.subplots()
    ax.plot(a, xs, 'b', label='First coordinate trajectory (real)')
    ax.plot(a[:-1:L], xs_m, 'g.', label='Noisy measurements', markersize=4.0)
    ax.plot(a[:-1:L], np.mean(x_tilde, axis=1), 'rx', label='CSMC output',
            markersize=4.0)
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
    #plot_samples_hist()
    plot_ekf()

if __name__ == "__main__":
    main()
