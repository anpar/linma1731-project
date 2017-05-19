from utils import simulate, measure, classical_smc

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math

PATH='../report/figures/'

def plot_trajectory(L, t_tot, dt, ts, xs_m, wxs, x, label, filename):
    a = np.arange(0, int(t_tot/dt) + 1, 1)
    fig, ax = plt.subplots()
    if xs_m is not None:
        ax.plot(a[::L], xs_m, 'g.', label='Noisy measurements', markersize=4.0)

    ax.plot(a[::L], wxs, 'rx', label='CSMC output', markersize=3.0)
    ax.plot(np.zeros(len(x[0, :])), x[0, :], 'b.',
           markersize=2.0, label='Particles')
    ax.plot(5/dt*np.ones(len(x[0, :])), x[int(5/ts), :], 'b.',
            markersize=2.0)
    ax.plot(10/dt*np.ones(len(x[0, :])), x[int(10/ts), :], 'b.',
           markersize=2.0)

    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    if filename is not None:
        fig.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)

    #plt.close(fig)

def plot_smc(n, filename):
    t_tot = 10
    dt = 0.001
    ts = 0.01
    L = int(ts/dt)

    xs_m = np.zeros((1001,))

    for i, x_m in enumerate(open('data.dat', 'r')):
        xs_m[i] = float(x_m)

    x_tilde, y_tilde, z_tilde, x, y, z, wxs = classical_smc(10, 28, 8/3, dt,
                                                            math.sqrt(0.01), np.eye(3),
                                                            0, math.sqrt(0.001), ts,
                                                            t_tot, xs_m, 1, n)

    # Trajectory of first coordinates
    plot_trajectory(L, t_tot, dt, ts, xs_m, wxs[:, 0], x, 'x-', filename[0])
    plot_trajectory(L, t_tot, dt, ts, None, wxs[:, 1], y, 'y-', filename[1])
    plot_trajectory(L, t_tot, dt, ts, None, wxs[:, 2], z, 'z-', filename[2])
    plt.show()

def main():
    filename = [None]*3
    plot_smc(n=100, filename=filename)

    """
    filename = ["x-trajectory-data.pdf", "y-trajectory-data.pdf",
                "z-trajectory-data.pdf"]
    plot_smc(n=1000, filename=filename)
    """

if __name__ == "__main__":
    main()
