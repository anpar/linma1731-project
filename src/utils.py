import math
import numpy as np

def get_init_pos(mu_0=1, sigma_0=math.sqrt(0.001)):
    """Return initial position of the Lorenz system.

    Arguments:
    mu_0 -- mean of the Gaussian distribution (default (1,1,1))
    sigma_0 -- standard deviation of the Gaussian distribution (default
    math.sqrt(0.001))
    """

    return np.random.normal(mu_0, sigma_0, 3)

def F(x, y, z, a=10, r=28, b=8/3, dt=0.001):
    """Compute the function F of the discrete-time version of the Lorenz
    system using first-order forward finite difference.

    Arguments:
    x, y, z -- system state vector (particle position)
    a, r, b -- system parameters (default 10, 28, 8/3)
    dt -- time step (default 0.001)
    """

    f1 = a*y*dt + (1-a*dt)*x
    f2 = x*(r-z)*dt + (1-dt)*y
    f3 = x*y*dt + (1-b*dt)*z

    return (f1, f2, f3)

def next_state_vector(x, y, z, a=10, r=28, b=8/3, dt=0.001, Gamma=np.eye(3)):
    """Return the state vector at instant k+1 from the one at instant k.
    A small Gaussian noise of variance equal to 10*eps is added on the
    dynamics.

    Arguments:
    x, y, z -- system state vector (particle position)
    a, r, b -- system parameters (default 10, 28, 8/3)
    dt -- time step (default 0.001)
    Gamma -- matrix multiplying the noise vector (default np.eye(3))
    """

    eps = np.finfo(np.float32).eps
    u = np.random.normal(0, 10*eps, 3)

    return F(x, y, z, a, r, b, dt) + Gamma.dot(u)

def simulate(t_tot, mu_0=1, sigma_0=math.sqrt(0.001), a=10, r=28, b=8/3, dt=0.001, Gamma=np.eye(3)):
    """Simulate the Lorenz.

    Arguments:
    t_tot -- simulation time (in seconds)
    mu_0 -- mean of the Gaussian distribution of the initial position (default (1,1,1))
    sigma_0 -- standard deviation of the Gaussian distribution of the initial
    position (default math.sqrt(0.001))
    a, r, b -- system parameters (default 10, 28, 8/3)
    dt -- time step (default 0.001)
    Gamma -- matrix multiplying the noise vector (default np.eye(3))
    """

    n_iter = int(t_tot/dt) + 1

    xs = np.empty(n_iter)
    ys = np.empty(n_iter)
    zs = np.empty(n_iter)
    x, y, z = get_init_pos(mu_0, sigma_0)

    for i in range(n_iter):
        xs[i], ys[i], zs[i] = x, y, z

        x, y, z = next_state_vector(x, y, z, a, r, b, dt, Gamma)

    return (xs, ys, zs)

def measure(xs, L, sigma_m=1):
    """Take noisy measurements with a sampling period ts
    from the proces xs.

    Arguments:
    xs -- first coordinate of the trajectory. Should be obtained like xs, _, _
    = simulate(t_tot, dt=dt)
    L -- downsampling factor (equal to the sampling period divided by the time
    step)
    sigma_m -- standard deviation of the measurement noise (default 1)
    """

    xs_m = xs[:-1:L]

    return xs_m + np.random.normal(0, sigma_m, len(xs_m))

