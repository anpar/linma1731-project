from utils import simulate, measure

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

PATH='../report/figures/'

def plot_3d_trajectory(export=False, filename='q2-3d-trajectory.pdf'):
    """Simulate the 3d-system for 50 seconds

    Arguments:
    export -- indicate if the plot should be PDF exported and saved (default
    False)
    filename -- tikz generate filename (default q2-3d-trajectory.pdf)
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
    t_tot = 50
    dt = 0.001
    ts = 0.01
    sigma_m = 1

    a = np.arange(0, int(t_tot/dt)+1, 1)
    b = np.arange(0, int(t_tot/dt), int(ts/dt))

    xs, _, _ = simulate(50)
    xs_m = measure(xs, 50, 0.001, 0.01, 1)

    fig, ax = plt.subplots()
    ax.plot(a, xs, 'b', label='First coordinate trajectory')
    ax.plot(b, xs_m, 'g.', label='Noisy measurements', markersize=4.0)
    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    plt.show()

    if export:
        fig.savefig(PATH + filename)

def main():
    plot_3d_trajectory(False)
    plot_mes_vs_real(False)

if __name__ == "__main__":
    main()
