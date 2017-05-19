from utils import simulate, measure, ekf2

from matplotlib import pyplot as plt

import numpy as np
import math, sys

PATH='../report/figures/'

def plot_ekf(export, filename):
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

    a = np.arange(0, int(t_tot/dt) + 1, 1)
    fig, ax = plt.subplots()
    ax.plot(a, xs, 'b', label='First coordinate trajectory (real)')
    ax.plot(a[::L], xs_m, 'g.', label='Noisy measurements', markersize=6.0)
    ax.plot(a[::L], mu[:, 0], 'rx', label='EKF output', markersize=4.0)
    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    if export:
        fig.savefig(PATH + filename)

    plt.show()

def main():
    plot_ekf(False, '')

if __name__ == "__main__":
    main()
