from utils import simulate, measure, ekf

from matplotlib import pyplot as plt

import numpy as np
import math

PATH='../report/figures/'

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
    plot_ekf()

if __name__ == "__main__":
    main()
