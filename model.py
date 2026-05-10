import numpy as np
from scipy.linalg import expm


def build_Aa(gamma: float, omega: float, g_phi: float, lam: float) -> np.ndarray:
    """
    Continuous-time augmented system matrix for:
        X = [q, p, phi]^T

    Model:
        q_dot   = -(gamma/2) q + omega p
        p_dot   = -omega q -(gamma/2) p + g_phi phi
        phi_dot = -lam phi + noise
    """
    return np.array([
        [-gamma / 2.0,  omega,        0.0],
        [-omega,       -gamma / 2.0,  g_phi],
        [0.0,           0.0,         -lam],
    ], dtype=float)


def build_Ca(kappa: float, theta: float) -> np.ndarray:
    """
    Homodyne measurement matrix:
        y = C_a X + v
    """
    return np.sqrt(kappa) * np.array([
        [np.cos(theta), np.sin(theta), 0.0]
    ], dtype=float)


def discretize_state_matrix(A: np.ndarray, dt: float) -> np.ndarray:
    """
    Exact discretization of the state matrix:
        F = expm(A dt)
    """
    return expm(A * dt)


def discretize_system_van_loan(A: np.ndarray, Qc: np.ndarray, dt: float):
    """
    Exact discretization of the pair (A, Qc) using the Van Loan method.

    Continuous-time system:
        x_dot = A x + w_c
    with:
        E[w_c(t) w_c(s)^T] = Qc delta(t-s)

    Discrete-time equivalent:
        x_{k+1} = F x_k + w_k
        E[w_k w_k^T] = Qd
    """
    n = A.shape[0]

    M = np.block([
        [-A, Qc],
        [np.zeros((n, n)), A.T]
    ]) * dt

    EM = expm(M)

    EM12 = EM[:n, n:]
    EM22 = EM[n:, n:]

    F = EM22.T
    Qd = F @ EM12

    # Symmetrize to reduce numerical asymmetry
    Qd = 0.5 * (Qd + Qd.T)

    return F, Qd


def build_continuous_process_covariance(
    sigma_q: float,
    sigma_p: float,
    q_phi: float,
) -> np.ndarray:
    """
    Continuous-time covariance density Qc.

    Interpretation:
    - sigma_q^2 : cavity q quadrature noise intensity
    - sigma_p^2 : cavity p quadrature noise intensity
    - q_phi     : OU phase diffusion intensity
    """
    return np.diag([
        sigma_q**2,
        sigma_p**2,
        q_phi
    ]).astype(float)


def rmse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x_true - x_est) ** 2)))