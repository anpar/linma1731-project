from utils import simulate, measure, ekf2

from matplotlib import pyplot as plt

import numpy as np
import math, sys

PATH='../report/figures/'

def plot_trajectory(L, t_tot, dt, xs, xs_m, mu, label, filename):
    a = np.arange(0, int(t_tot/dt) + 1, 1)
    fig, ax = plt.subplots()
    ax.plot(a, xs, 'b', label=label + 'coordinate trajectory (real)')
    if xs_m is not None:
        ax.plot(a[::L], xs_m, 'g.', label='Noisy measurements', markersize=4.0)

    ax.plot(a[::L], mu, 'rx', label='EKF output', markersize=3.0)

    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    if filename is not None:
        fig.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)

def plot_ekf(filename):
    t_tot = 16
    ts = 0.01
    dt = 0.001
    L = int(ts/dt)
    mu_0 = 1
    sigma_0 = math.sqrt(0.001)
    sigma_u = math.sqrt(0.01)
    sigma_m = math.sqrt(1)
    a, r, b = 10, 28, 8/3
    Gamma = np.eye(3)

    xs, ys, zs = simulate(t_tot, mu_0, sigma_0, a, r, b, dt, sigma_u, Gamma)
    xs_m = measure(xs, L, sigma_m)

    mu, cov = ekf2(a, r, b, dt, sigma_u, Gamma, mu_0, sigma_0, ts, t_tot, xs_m,
                  sigma_m)

    print(cov[int(5/ts)][0,0])

    plot_trajectory(L, t_tot, dt, xs, xs_m, mu[:, 0], 'x', filename[0])
    plot_trajectory(L, t_tot, dt, ys, None, mu[:, 1], 'y', filename[1])
    plot_trajectory(L, t_tot, dt, zs, None, mu[:, 2], 'z', filename[2])

    # Error function
    fig, ax = plt.subplots()
    x_real = np.empty((int(t_tot/ts)+1, 3))
    x_real[:, 0] = xs[::L]
    x_real[:, 1] = ys[::L]
    x_real[:, 2] = zs[::L]

    a = np.arange(0, int(t_tot/dt) + 1, 1)
    err = np.linalg.norm(x_real - mu, axis=1)
    plt.plot(a[::L], err, 'b', label="Global error")
    plt.axhline(np.mean(err), color='b', linestyle='dashed', label="Mean global error")
    err_x = np.abs(x_real[:, 0] - mu[:, 0])
    plt.axhline(np.mean(err_x), color='g', linestyle='dashed', label="Mean error on x")

    legend = ax.legend(loc='upper right')
    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    if filename[3] is not None:
        fig.savefig(PATH + filename[3], bbox_inches='tight', pad_inches=0)

    plt.show()

def main():
    filename = [None, None, None, None]
    plot_ekf(filename)

if __name__ == "__main__":
    main()
