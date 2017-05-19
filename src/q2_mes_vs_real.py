from utils import simulate, measure

from matplotlib import pyplot as plt

import numpy as np
import math

PATH='../report/figures/'

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
    xs, ys, zs = simulate(a=10, r=28, b=8/3, mu_0=(1,1,1),
                          sigma_0=math.sqrt(0.001), dt=dt,
                          sigma_u=math.sqrt(0.0000001), Gamma=np.eye(3),
                          t_tot=t_tot)
    xs_m = measure(xs, L=L, sigma_m=1)

    fig, ax = plt.subplots()
    a = np.arange(0, int(t_tot/dt) + 1, 1)
    ax.plot(a, xs, 'b', label='First coordinate trajectory')
    ax.plot(a[:-1:L], xs_m, 'g.', label='Noisy measurements', markersize=4.0)
    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    plt.show()

    if export:
        fig.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)

def main():
    plot_mes_vs_real(False)

if __name__ == "__main__":
    main()
