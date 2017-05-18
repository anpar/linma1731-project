import sys
import math
import numpy as np
import scipy.stats

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
    u = np.random.normal(0, 100000*eps, 3)
    u = np.random.normal(0, 0.35, 3)

    return F(x, y, z, a, r, b, dt) + Gamma.dot(u)

def simulate(t_tot, mu_0=1, sigma_0=math.sqrt(0.001), a=10, r=28, b=8/3, dt=0.001, Gamma=np.eye(3)):
    """Simulate the Lorenz system.

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

def print_progress(perc):
    sys.stdout.write("\r%0.2f%%" % perc)
    sys.stdout.flush()

def next_state_vector_L(x, y, z, L, a=10, r=28, b=8/3, dt=0.001, Gamma=np.eye(3)):
    """ Apply next_state_vector L times."""

    x_cur = np.copy(x)
    y_cur = np.copy(y)
    z_cur = np.copy(z)

    for i in range(L):
        x_cur, y_cur, z_cur = next_state_vector(x_cur, y_cur, z_cur, a, r, b,
                                                dt, Gamma)

    return x_cur, y_cur, z_cur

def classical_smc(xs_m, t_tot, L, n=100, mu_0=1, sigma_0=math.sqrt(0.001), sigma_m=1, dt=0.001,
                  ts=0.01, Gamma=np.eye(3)):
    """Classical Sequential Monte Carlo."""

    n_iter = len(xs_m)
    assert n_iter == int(t_tot/ts)

    x_tilde = np.empty((n_iter, n))
    y_tilde = np.empty((n_iter, n))
    z_tilde = np.empty((n_iter, n))
    weights = np.empty(n)

    # Generates initial sample sets
    x_tilde[0, :] = np.random.normal(mu_0, sigma_0, n)
    y_tilde[0, :] = np.random.normal(mu_0, sigma_0, n)
    z_tilde[0, :] = np.random.normal(mu_0, sigma_0, n)

    wxs = np.zeros((3, n_iter))

    # Loop on time
    for t in range(1, n_iter):
        print_progress(t/n_iter * 100.0)

        # Prediction
        for i in range(n):
            x_tilde[t, i], y_tilde[t, i], z_tilde[t, i] = \
            next_state_vector_L(x_tilde[t-1, i], y_tilde[t-1, i],
                                z_tilde[t-1, i], L)
        # Correction
        for i in range(n):
            weights[i] = scipy.stats.norm.pdf(x_tilde[t, i], xs_m[t],
                                              sigma_m)

        weights /= sum(weights)

        for i in range(n):
            wxs[:, t] += weights[i] * np.array([x_tilde[t, i], y_tilde[t, i],
                                          z_tilde[t, i]])

        # Resample the particles according to the weights
        ind_sample = np.random.choice(np.arange(n), n, True, weights)

        x_tilde[t, :] = x_tilde[t, ind_sample]
        y_tilde[t, :] = y_tilde[t, ind_sample]
        z_tilde[t, :] = z_tilde[t, ind_sample]

    return x_tilde, y_tilde, z_tilde, wxs

def next_state_jacobian(x, y, z, a=10, r=28, b=8/3, dt=0.001):
    j = [
        [1. - a*dt, a*dt,  0.],
        [(r-z)*dt,  1-dt, -x*dt],
        [y*dt,      x*dt,  1-b*dt]
    ]

    return np.matrix(j)

def ekf(xs_m, t_tot, L, n=100, mu_0=1, sigma_0=math.sqrt(0.001), sigma_m=1, dt=0.001,
        ts=0.01, Gamma=np.eye(3)):
    """Extended Kalman Filter"""

    n_iter = len(xs_m)
    assert n_iter == int(t_tot/ts)

    x_predicted = [np.zeros((3, 1)) for _ in range(n_iter+1)]
    x_updated   = [np.zeros((3, 1)) for _ in range(n_iter)]

    cov_predicted = [np.zeros((3,3)) for _ in range(n_iter+1)]
    cov_updated   = [np.zeros((3,3)) for _ in range(n_iter)]

    # Initializing
    x_predicted[0] = np.matrix([[mu_0], [mu_0], [mu_0]])
    cov_predicted[0] = Gamma * sigma_0**2

    H = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    eps = np.finfo(np.float32).eps
    sigma_u = 100000*eps
    sigma_u = 0.35 # TODO works best with this value ...

    # Loop on time
    for t in range(n_iter):
        #print_progress(t/n_iter * 100.0)

        ## Update for t

        # Computing K_t
        #K = cov_predicted[t] @ np.transpose(H) @ np.linalg.inv(H @ cov_predicted[t] @ np.transpose(H) + H*sigma_m**2)
        K_x = cov_predicted[t][0,0] / (cov_predicted[t][0,0] + sigma_m**2)
        K = np.array([ [K_x, 0, 0], [0, 0, 0], [0, 0, 0] ])

        # Updating x_t|t
        x_updated[t] = x_predicted[t] + K @ (np.matrix([ [xs_m[t] - x_predicted[t][0,0]], [0], [0]]))
        # Updating P_t|t
        cov_updated[t] = cov_predicted[t] - K @ H @ cov_predicted[t]

        ## Prediction for t+1

        # Computing x_(t+1)|t
        x = x_updated[t][0,0]
        y = x_updated[t][1,0]
        z = x_updated[t][2,0]

        x_tilde, y_tilde, z_tilde = next_state_vector(x, y, z, Gamma=np.zeros((3,3)))
        x_predicted[t+1] = np.array([[x_tilde], [y_tilde], [z_tilde]])

        # Computing P_(t+1)|t
        F = next_state_jacobian(x,y,z)
        cov_predicted[t+1] = F @ cov_updated[t] @ np.transpose(F) + Gamma* sigma_u**2

    return x_updated
