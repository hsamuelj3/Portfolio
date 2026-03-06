import casadi as ca
import numpy as np

M, m, l, g = 1.0, 0.1, 0.5, 9.81
dt = 0.05
N  = 20  # horizon

def cartpole_dynamics(s, u):
    """Nonlinear cartpole RHS — returns x[k+1] via Euler step."""
    p, p_dot, theta, theta_dot = s[0], s[1], s[2], s[3]
    F = u[0]

    sin_t = ca.sin(theta)
    cos_t = ca.cos(theta)
    denom = M + m * sin_t**2

    p_ddot     = (F + m*l*theta_dot**2*sin_t - m*g*sin_t*cos_t) / denom
    theta_ddot = ((M+m)*g*sin_t - cos_t*(F + m*l*theta_dot**2*sin_t)) / (l * denom)

    # Euler integration
    return ca.vertcat(
        p     + dt * p_dot,
        p_dot + dt * p_ddot,
        theta + dt * theta_dot,
        theta_dot + dt * theta_ddot
    )

def solve_mpc(x0, x_ref):
    opti = ca.Opti()

    X = opti.variable(4, N+1)   # states
    U = opti.variable(1, N)     # inputs (force)

    # Cost weights
    Q = np.diag([1.0, 0.1, 10.0, 0.1])  # penalize θ error most
    R = np.array([[0.01]])

    cost = 0
    opti.subject_to(X[:, 0] == x0)

    for k in range(N):
        dx = X[:, k] - x_ref
        cost += dx.T @ Q @ dx + U[:, k].T @ R @ U[:, k]

        # Dynamics constraint
        opti.subject_to(X[:, k+1] == cartpole_dynamics(X[:, k], U[:, k]))

        # Force limits
        opti.subject_to(opti.bounded(-10, U[:, k], 10))

    # Terminal cost
    dx_N = X[:, N] - x_ref
    cost += dx_N.T @ Q @ dx_N

    opti.minimize(cost)
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})

    try:
        sol = opti.solve()
        return float(sol.value(U[:, 0]))
    except:
        return 0.0  # fallback if solver fails

# Example usage
x0  = np.array([0.0, 0.0, 0.15, 0.0])  # slight tilt
ref = np.array([0.0, 0.0, 0.0,  0.0])  # upright, centered
u   = solve_mpc(x0, ref)
print(f"Control force: {u:.3f} N")