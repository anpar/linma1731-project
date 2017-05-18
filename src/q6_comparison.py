from utils import simulate, measure, classical_smc, ekf
from matplotlib import pyplot as plt

import math
import numpy as np

a = 10
r = 28
b = 8/3
t_tot = 16
mu_0 = 1
dt = 0.001
sigma_u = math.sqrt(0.01)
sigma_0 =math.sqrt(0.001)
ts = 0.01
Gamma = np.eye(3)
sigma_m = 1
n = 100
L = int(ts / dt)

xs, ys, zs = simulate(t_tot, mu_0, sigma_0, a, r, b,
              dt, sigma_u, Gamma,)
xs_m = measure(xs, L, sigma_m)

x_tilde, y_tilde, z_tilde, x, y, z, wxs = classical_smc(a, r, b, dt,
                                                            sigma_u, Gamma,
                                                            mu_0, sigma_0, ts,
                                                            t_tot, xs_m,
                                                            sigma_m, n)

# Error function
fig, ax = plt.subplots()
x_real = np.empty((int(t_tot/ts), 3))
x_real[:, 0] = xs[:-1:L]
x_real[:, 1] = ys[:-1:L]
x_real[:, 2] = zs[:-1:L]

a = np.arange(0, int(t_tot/dt) + 1, 1)
err = np.linalg.norm(x_real - wxs, axis=1)
plt.plot(a[:-1:L], err, 'b', label="Global error")
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

#if filename[2] is not None:
#    fig.savefig(PATH + filename[2])

plt.show()

a = np.arange(0, int(t_tot/dt) + 1, 1)
fig, ax = plt.subplots()
ax.plot(a, xs, 'b',label='x-coordinate trajectory (real)')
if xs_m is not None:
    ax.plot(a[:-1:L], xs_m, 'g.', label='Noisy measurements', markersize=4.0)

ax.plot(a[:-1:L], wxs[:,0], 'rx', label='CSMC output',
        markersize=3.0)

legend = ax.legend(loc='upper right')

for label in legend.get_texts():
     label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)

plt.show()


xs, ys, zs = xs[::L], ys[::L], zs[::L]
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

plt.show()


# Error function
fig, ax = plt.subplots()
x_real = np.empty((int(t_tot/ts), 3))
x_real[:, 0] = xs[:-1:]
x_real[:, 1] = ys[:-1:]
x_real[:, 2] = zs[:-1:]
print(x_real)

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

if filename is not None:
    fig.savefig(PATH + filename[2])

plt.show()

