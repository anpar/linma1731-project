from utils import simulate, measure, classical_smc

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math

PATH='../report/figures/'

def plot_hist(x, xs, dt, ts, filename):
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.hist(x[int(5/ts), :], color='b', align='mid', alpha=0.5)
    plt.axvline(xs[int(5/dt)], color='g', linestyle='dashed', linewidth=2)

    plt.subplot(1, 3, 2)
    plt.hist(x[int(10/ts), :], color='b', align='mid', alpha=0.5)
    plt.axvline(xs[int(10/dt)], color='g', linestyle='dashed', linewidth=2)

    plt.subplot(1, 3, 3)
    plt.hist(x[int(15/ts), :], color='b', align='mid', alpha=0.5)
    plt.axvline(xs[int(15/dt)], color='g', linestyle='dashed', linewidth=2)

    if filename is not None:
        fig.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)

def plot_particles(x_tilde, y_tilde, z_tilde, x, y, z, xs_m, wxs, t, ts,
                   filename):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot(x_tilde[int(t/ts), :], y_tilde[int(t/ts), :], z_tilde[int(t/ts), :],
           'b.', label='Particles before resampling', markersize=7.0)
    ax.plot(x[int(t/ts), :], y[int(t/ts), :], z[int(t/ts), :],
           'rx', label='Particles after resampling', alpha=0.5, markersize=8.0)
    ax.scatter(wxs[int(t/ts)][0], wxs[int(t/ts)][1], wxs[int(t/ts)][2], color='k',
               label="Weighted average")

    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    if filename is not None:
        fig.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)

    #plt.close(fig)

def plot_trajectory(L, t_tot, dt, xs, xs_m, wxs, label, filename):
    a = np.arange(0, int(t_tot/dt) + 1, 1)
    fig, ax = plt.subplots()
    ax.plot(a, xs, 'b', label=label + 'coordinate trajectory (real)')
    if xs_m is not None:
        ax.plot(a[::L], xs_m, 'g.', label='Noisy measurements', markersize=4.0)

    ax.plot(a[::L], wxs, 'rx', label='CSMC output',
            markersize=3.0)

    legend = ax.legend(loc='upper right')

    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    if filename is not None:
        fig.savefig(PATH + filename, bbox_inches='tight', pad_inches=0)

    #plt.close(fig)

def plot_smc(a, r, b, dt, ts, t_tot, mu_0, sigma_0, sigma_u, Gamma,
                      sigma_m, n, filename):

    L = int(ts/dt)

    xs, ys, zs = simulate(t_tot, mu_0, sigma_0, a, r, b,
                          dt, sigma_u, Gamma,)
    xs_m = measure(xs, L, sigma_m)

    x_tilde, y_tilde, z_tilde, x, y, z, wxs = classical_smc(a, r, b, dt,
                                                            sigma_u, Gamma,
                                                            mu_0, sigma_0, ts,
                                                            t_tot, xs_m,
                                                            sigma_m, n)

    # Histograms
    plot_hist(x, xs, dt, ts, filename[0])
    plot_hist(y, ys, dt, ts, filename[1])

    # Error function
    fig, ax = plt.subplots()
    x_real = np.empty((int(t_tot/ts)+1, 3))
    x_real[:, 0] = xs[::L]
    x_real[:, 1] = ys[::L]
    x_real[:, 2] = zs[::L]

    a = np.arange(0, int(t_tot/dt) + 1, 1)
    err = np.linalg.norm(x_real - wxs, axis=1)
    plt.plot(a[::L], err, 'b', label="Global error")
    plt.axhline(np.mean(err), color='b', linestyle='dashed',
                label="Mean global error")
    err_x = np.abs(x_real[:, 0] - wxs[:, 0])
    plt.axhline(np.mean(err_x), color='g', linestyle='dashed',
                label="Mean error on x")
    #plt.ylim(0, 6)

    legend = ax.legend(loc='upper right')
    for label in legend.get_texts():
         label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    if filename[2] is not None:
        fig.savefig(PATH + filename[2], bbox_inches='tight', pad_inches=0)

    plt.close(fig)

    # Particles
    plot_particles(x_tilde, y_tilde, z_tilde, x, y, z, xs_m, wxs, 5, ts,
                   filename[3])
    plot_particles(x_tilde, y_tilde, z_tilde, x, y, z, xs_m, wxs, 15, ts,
                   filename[4])

    # Trajectory of first coordinates
    plot_trajectory(L, t_tot, dt, xs, xs_m, wxs[:, 0], 'x-', filename[5])
    plot_trajectory(L, t_tot, dt, ys, None, wxs[:, 1], 'y-', filename[6])
    plot_trajectory(L, t_tot, dt, zs, None, wxs[:, 2], 'z-', filename[7])

    plt.show()

def main():
    a = 10
    r = 28
    b = 8/3

    sigma_u = math.sqrt(0.01)

    filename = [None]*8

    plot_smc(a=a, r=r, b=b, dt=0.001, ts=0.01, t_tot=16, mu_0=1,
             sigma_0=math.sqrt(64), sigma_u=sigma_u,
             Gamma=np.eye(3), sigma_m=math.sqrt(1), n=100,
             filename=filename)

    plot_smc(a=a, r=r, b=b, dt=0.001, ts=0.01, t_tot=16, mu_0=1,
             sigma_0=math.sqrt(0.001), sigma_u=sigma_u,
             Gamma=np.eye(3), sigma_m=math.sqrt(0.001), n=100,
             filename=filename)

if __name__ == "__main__":
    main()
