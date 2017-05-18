from utils import simulate, measure, ekf

from matplotlib import pyplot as plt

import numpy as np
import math

PATH='../report/figures/'

def plot_ekf(export=False, filename='q6-samples-ekf.pdf'):
    t_tot = 16
    ts = 0.01
    mu_0, sigma_0 = 1, math.sqrt(0.001)
    sigma_u = math.sqrt(0.01)
    sigma_m = math.sqrt(1)
    a, r, b = 10, 28, 8/3
    Gamma = np.eye(3)

    xs, ys, zs = simulate(t_tot, mu_0, sigma_0, a, r, b, ts, sigma_u, Gamma)
    xs_m = measure(xs, 1, sigma_m)

    x_tilde = np.array([X[0] for X in ekf(xs_m, t_tot, a, r, b, mu_0, sigma_0, sigma_m, sigma_u, ts, Gamma)])

    a = np.arange(0, len(xs_m) + 1, 1)
    fig, ax = plt.subplots()
    ax.plot(a, xs, 'b', label='First coordinate trajectory (real)')
    ax.plot(a[:-1], xs_m, 'g.', label='Noisy measurements', markersize=4.0)
    ax.plot(a[:-1], np.mean(x_tilde, axis=1), 'rx', label='EKF output',
            markersize=4.0)
    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    plt.show()

def main():
    plot_ekf()

if __name__ == "__main__":
    main()
