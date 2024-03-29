import sys
import math
import numpy as np
import scipy.stats

"""
    IMPORTANT NOTE: all sigma_X are STANDARD DEVIATION
    (and not variances)
"""

def get_init_pos(mu_0, sigma_0):
    """Return initial position of the Lorenz system.

    Arguments:
    mu_0 -- mean of the Gaussian distribution
    sigma_0 -- standard deviation of the Gaussian distribution
    """

    return np.random.normal(mu_0, sigma_0, 3)

def F(x, y, z, a, r, b, dt):
    """Compute the function F of the discrete-time version of the Lorenz
    system using first-order forward finite difference.

    Arguments:
    x, y, z -- system state vector (particle position)
    a, r, b -- system parameters
    dt -- time step
    """

    f1 = a*y*dt + (1-a*dt)*x
    f2 = x*(r-z)*dt + (1-dt)*y
    f3 = x*y*dt + (1-b*dt)*z

    return (f1, f2, f3)

def next_state_vector(x, y, z, a, r, b, dt, sigma_u, Gamma):
    """Return the state vector at instant k+1 from the one at instant k.
    A small Gaussian noise of variance equal to sigma_u is added on the
    dynamics.

    Arguments:
    x, y, z -- system state vector (particle position)
    a, r, b -- system parameters
    dt -- time step
    Gamma -- matrix multiplying the noise vector
    """

    u = np.random.normal(0, sigma_u, 3)

    return F(x, y, z, a, r, b, dt) + Gamma.dot(u)

def simulate(t_tot, mu_0, sigma_0, a, r, b, dt, sigma_u, Gamma):
    """Simulate the Lorenz system.

    Arguments:
    t_tot -- simulation time (in seconds)
    mu_0 -- mean of the Gaussian distribution of the initial position
    sigma_0 -- standard deviation of the Gaussian distribution of the initial
    position
    a, r, b -- system parameters
    dt -- time step
    Gamma -- matrix multiplying the noise vector
    """

    n_iter = int(t_tot/dt) + 1

    xs = np.empty(n_iter)
    ys = np.empty(n_iter)
    zs = np.empty(n_iter)
    x, y, z = get_init_pos(mu_0, sigma_0)

    for i in range(n_iter):
        xs[i], ys[i], zs[i] = x, y, z

        x, y, z = next_state_vector(x, y, z, a, r, b, dt, sigma_u, Gamma)

    return (xs, ys, zs)

def measure(xs, L, sigma_m):
    """Take noisy measurements with a sampling period ts
    from the proces xs.

    Arguments:
    xs -- first coordinate of the trajectory. Should be obtained like xs, _, _
    = simulate(t_tot, dt=dt)
    L -- downsampling factor (equal to the sampling period divided by the time
    step)
    sigma_m -- standard deviation of the measurement noise
    """

    xs_m = xs[::L]
    n = np.random.normal(0, sigma_m, len(xs_m))

    return xs_m + n

def print_progress(perc):
    sys.stdout.write("\r%0.2f%%" % perc)
    sys.stdout.flush()

def next_state_vector_L(x, y, z, L, a, r, b, dt, sigma_u, Gamma):
    """ Apply next_state_vector L times."""

    x_cur = np.copy(x)
    y_cur = np.copy(y)
    z_cur = np.copy(z)

    for i in range(L):
        x_cur, y_cur, z_cur = next_state_vector(x_cur, y_cur, z_cur, a, r, b,
                                                dt, sigma_u, Gamma)

    return x_cur, y_cur, z_cur

def classical_smc(a, r, b, dt, sigma_u, Gamma, mu_0, sigma_0, ts,
                  t_tot, xs_m, sigma_m, n, particles_at=None):
    """Classical Sequential Monte Carlo."""

    L = int(ts/dt)

    n_iter = len(xs_m)
    assert n_iter == int(t_tot/ts) + 1

    # Particles
    x = np.zeros((n_iter, n))
    y = np.zeros((n_iter, n))
    z = np.zeros((n_iter, n))
    # Predictions
    x_tilde = np.zeros((n_iter, n))
    y_tilde = np.zeros((n_iter, n))
    z_tilde = np.zeros((n_iter, n))
    # Importance resampling weights
    weights = np.zeros(n)

    # Generates initial sample sets
    x[0, :] = np.random.normal(mu_0, sigma_0, n)
    y[0, :] = np.random.normal(mu_0, sigma_0, n)
    z[0, :] = np.random.normal(mu_0, sigma_0, n)

    wxs = np.zeros((n_iter, 3))

    # Loop on time
    for t in range(1, n_iter):
        print_progress(t/n_iter * 100.0)

        # Prediction
        for i in range(n):
            x_tilde[t, i], y_tilde[t, i], z_tilde[t, i] = \
            next_state_vector_L(x[t-1, i], y[t-1, i], z[t-1, i], L, a, r, b,
                               dt, sigma_u, Gamma)

        # Computing weights for importance resampling
        for i in range(n):
            weights[i] = scipy.stats.norm.pdf(xs_m[t] - x_tilde[t, i], 0, sigma_m)

        weights = weights/sum(weights)

        # Weighted average, used as estimate
        for i in range(n):
            wxs[t, :] += weights[i] * np.array([x_tilde[t, i], y_tilde[t, i],
                                                z_tilde[t, i]])

        # Resample the particles according to the weights
        ind_sample = np.random.choice(n, n, replace=True, p=weights)

        x[t, :] = x_tilde[t, ind_sample]
        y[t, :] = y_tilde[t, ind_sample]
        z[t, :] = z_tilde[t, ind_sample]

        if particles_at == t:
            return [ x[t,:], y[t, :], z[t, :] ]

    return x_tilde, y_tilde, z_tilde, x, y, z, wxs

def jacobian(x, y, z, a, r, b, dt):
    """Return the Jacobian of the function F."""

    J = [[1. - a*dt, a*dt, 0.],
        [(r-z)*dt,  1-dt, -x*dt],
        [y*dt, x*dt, 1-b*dt]]

    return np.array(J)

def ekf(a, r, b, dt, sigma_u, Gamma, mu_0, sigma_0, ts,
         t_tot, xs_m, sigma_m):
    """Extended Kalman Filter (EKF)."""

    L = int(ts/dt)

    n_iter = len(xs_m)
    assert n_iter == int(t_tot/ts) + 1

    Q = (sigma_u)**2 * Gamma
    R = (sigma_m)**2
    P0 = (sigma_0)**2 * np.eye(3)

    # Predicted mean/covariance
    mu_pred = np.zeros((n_iter, 3))
    cov_pred = np.zeros((n_iter, 3, 3))

    # Updated mean/covariance
    mu = np.zeros((n_iter, 3))
    cov = np.zeros((n_iter, 3, 3))

    # Initial guess for t = 0
    mu_pred[0] = np.array([mu_0]*3)
    cov_pred[0] = P0

    for t in range(n_iter-1):
        # Correction using measurements at time t
        K = cov_pred[t][:, 0] / (R + cov_pred[t][0, 0])
        mu[t] = mu_pred[t] + K * (xs_m[t] - mu_pred[t, 0])
        cov[t] = cov_pred[t] - np.outer(K, cov_pred[t][0, :])

        # Prediction for time t+1
        x, y, z = mu[t]
        mu_pred[t+1, 0], mu_pred[t+1, 1], mu_pred[t+1, 2] = \
        next_state_vector_L(x, y, z, L, a, r, b, dt, sigma_u, Gamma)

        c = cov[t]
        # We must apply the Jacobian L times
        for i in range(L):
            A = jacobian(x, y, z, a, r, b, dt)
            c = (A.dot(c)).dot(A.transpose()) + (Gamma.dot(Q)).dot(Gamma)

        cov_pred[t+1] = c

    return mu, cov
