import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp


def pendulum_system(t, state):
    """
    Ground-truth pendulum dynamics.

    state[0] = angle (theta)
    state[1] = angular velocity (omega)
    """

    theta, omega = state

    g = 9.81
    length = 1.0

    dtheta_dt = omega
    domega_dt = -(g / length) * np.sin(theta)

    return [dtheta_dt, domega_dt]


def generate_pendulum_data():
    """Generate a simulated pendulum trajectory."""

    t = np.linspace(0, 10, 1000)

    initial_state = [
        np.radians(30),
        0.0
    ]

    solution = solve_ivp(
        pendulum_system,
        [t[0], t[-1]],
        initial_state,
        t_eval=t
    )

    theta = solution.y[0]
    omega = solution.y[1]

    return t, theta, omega


def discover_dynamics(t, theta, omega):
    """Use SINDy to discover the governing dynamics."""

    X = np.column_stack([
        theta,
        omega
    ])

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.05),
        feature_library=ps.PolynomialLibrary(degree=3)
    )

    model.fit(X, t=t)

    return model