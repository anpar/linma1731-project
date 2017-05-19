from utils import simulate

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
    xs, ys, zs = simulate(a=10, r=28, b=8/3, mu_0=(1,1,1),
                          sigma_0=math.sqrt(0.001), dt=0.001,
                          sigma_u=math.sqrt(0.0000001), Gamma=np.eye(3),
                          t_tot=50)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(xs[0], ys[0], zs[0], color='g')
    ax.plot(xs, ys, zs)
    plt.show()

    if export:
        fig.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)

def main():
    plot_3d_trajectory(False)

if __name__ == "__main__":
    main()
