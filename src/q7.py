from utils import simulate, measure, ekf, classical_smc

from matplotlib import pyplot as plt

import numpy as np
import math

PATH='../report/figures/'

t_tot = 16
ts = 0.01
dt = 0.001
mu_0, sigma_0 = 1, math.sqrt(0.001)
sigma_u = math.sqrt(0.01)
sigma_m = math.sqrt(1)
a, r, b = 10, 28, 8/3
Gamma = np.eye(3)
n = 100
dimensions = ('x','y','z')

def ekf_distribs(xs_m):
    pos, cov = ekf(xs_m, t_tot, a, r, b, mu_0, sigma_0, sigma_m, sigma_u, ts,
                   Gamma, cov_at=5/ts)

    print(pos.shape)
    distribs = [None]*3
    for k in range(3):
        distribs[k] = np.random.normal(pos[k, 0], math.sqrt(cov[k,k]), n)

    return distribs

def csmc_distribs(xs_m):
    L = int(ts/dt)

    particles = classical_smc(a, r, b, dt, sigma_u, Gamma,
                              mu_0, sigma_0, ts, t_tot, xs_m,
                              sigma_m, n, particles_at=5/ts)

    return particles

def main():
    L = int(ts/dt)
    xs, ys, zs = simulate(t_tot, mu_0, sigma_0, a, r, b,
                          dt, sigma_u, Gamma,)
    xs_m = measure(xs, L, sigma_m)

    distribs_ekf = ekf_distribs(xs_m[:-1])
    distribs_csmc = csmc_distribs(xs_m)

    for k in range(3):
        C_ekf = np.sort(distribs_ekf[k])
        C_csmc = np.sort(distribs_csmc[k])
        R = np.arange(n)/float(n)

        fig, ax = plt.subplots()
        ax.plot(C_ekf, R, label="{} distrib. from EKF".format(dimensions[k]))
        ax.plot(C_csmc, R, label="{} distrib. from CSMC".format(dimensions[k]))

        legend = ax.legend(loc='upper right')

        for label in legend.get_texts():
             label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)

        fig.savefig(PATH + "distrib-{}.pdf".format(dimensions[k]))

        plt.show()

if __name__ == "__main__":
    main()
