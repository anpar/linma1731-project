import math
import numpy as np

def get_init_pos(mu_0=1, sigma_0=math.sqrt(0.001)):
    """Return initial position of the Lorenz system."""

    return np.random.normal(mu_0, sigma_0, 3)

def F(x, y, z, a, r, b, dt):
    """Compute the function F of the discrete-time version of the Lorenz
    system using first-order forward finite difference."""

    f1 = a*y*dt + (1-a*dt)*x
    f2 = x*(r-z)*dt + (1-dt)*y
    f3 = x*y*dt + (1-dt)*z

    return (f1, f2, f3)

def next_state_vector(x, y, z, a=10, r=28, b=8/3, dt=0.001, Gamma=np.eye(3)):
    """Return the state vector at instant k+1 from the one at instant k.

    Arguments:
    x -- system state vector (3-uple)
    a, r, b -- system paramaters (default 10, 28, 8/3)
    dt -- time step (default 0.001)
    Gamma -- matrix multiplying the noise vector (default np.eye(3))
    """

    eps = np.finfo(np.float32).eps
    u = np.random.normal(0, 10*eps, 3)

    return F(x, y, z, a, r, b, dt) + Gamma.dot(u)

def simulate(t_tot, dt=0.001):
    """Simulate the Lorenz system during a given amount of time"""

    n_iter = int(t_tot/dt) + 1

    xs = np.empty(n_iter)
    ys = np.empty(n_iter)
    zs = np.empty(n_iter)
    x, y, z = get_init_pos()

    for i in range(n_iter):
        xs[i], ys[i], zs[i] = x, y, z

        x, y, z = next_state_vector(x, y, z, dt=dt)

    return (xs, ys, zs)

def measure(xs, t_tot, dt, ts, sigma_m):
    assert len(xs) == t_tot/dt + 1

    L = ts/dt # downsampling factor
    idx = [int(i*L) for i in range(int(len(xs)/L))]

    return xs[idx] + np.random.normal(0, sigma_m, len(idx))

