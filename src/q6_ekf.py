from utils import simulate, measure, ekf

from matplotlib import pyplot as plt

import numpy as np
import math, sys

PATH='../report/figures/'

def plot_ekf(filename=[None, None]):
    t_tot = 16
    ts = 0.01
    mu_0, sigma_0 = 1, math.sqrt(0.001)
    sigma_u = math.sqrt(0.01)
    sigma_m = math.sqrt(1)
    a, r, b = 10, 28, 8/3
    Gamma = np.eye(3)

    xs, ys, zs = simulate(t_tot, mu_0, sigma_0, a, r, b, ts, sigma_u, Gamma)
    xs_m = measure(xs, 1, sigma_m)[:-1]

    wxs = np.matrix([[X[0,0], X[1,0], X[2,0]] for X in ekf(xs_m, t_tot, a, r, b, mu_0, sigma_0, sigma_m, sigma_u, ts, Gamma)])
    x_tilde = wxs[:,0]

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

    if filename[0] is not None:
        fig.savefig(PATH + filename[0])

    plt.show()


    # Error function
    fig, ax = plt.subplots()
    x_real = np.empty((int(t_tot/ts), 3))
    x_real[:, 0] = xs[:-1:]
    x_real[:, 1] = ys[:-1:]
    x_real[:, 2] = zs[:-1:]

    a = np.arange(0, int(t_tot/ts) + 1, 1)
    err = np.linalg.norm(x_real - wxs, axis=1)
    plt.plot(a[:-1], err, 'b', label="Global error")
    plt.axhline(np.mean(err), color='b', linestyle='dashed',
                label="Mean global error")
    err_x = np.abs(x_real[:, 0] - wxs[:, 0])
    plt.axhline(np.mean(err_x), color='g', linestyle='dashed',
                label="Mean error on x")

    legend = ax.legend(loc='upper right')
    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    if filename[1] is not None:
        fig.savefig(PATH + filename[1])

    plt.show()
    #plt.close(fig)


def main():
    if len(sys.argv) < 2 or (len(sys.argv) == 2 and sys.argv[1] not in ("good", "bad")):
        print("Please specify which dataset you want to write: 'good' or 'bad'")
        print(">>> python q6_ekf.py xxxx")
        sys.exit(1)

    plot_ekf(filename=['ekf-{}-x-trajectory.pdf'.format(sys.argv[1]), 'ekf-{}-error.pdf'.format(sys.argv[1])])

if __name__ == "__main__":
    main()
