import numpy as np
from scipy.linalg import expm


def build_qco_augmented_A(
    gamma_s: float,
    omega_s: float,
    gamma_o: float,
    omega_o: float,
    g_phi: float,
    lam: float,
    k_so: float,
    k_os: float,
) -> np.ndarray:
    """
    Augmented linear QCO model:
        X = [q_s, p_s, q_o, p_o, phi]^T
    """

    A_s = np.array([
        [-gamma_s / 2.0,  omega_s],
        [-omega_s,       -gamma_s / 2.0],
    ], dtype=float)

    A_o = np.array([
        [-gamma_o / 2.0,  omega_o],
        [-omega_o,       -gamma_o / 2.0],
    ], dtype=float)

    # simple linear couplings
    A_so = np.array([
        [0.0, 0.0],
        [k_so, 0.0],
    ], dtype=float)

    A_os = np.array([
        [0.0, 0.0],
        [k_os, 0.0],
    ], dtype=float)

    B_s = np.array([[0.0], [g_phi]], dtype=float)

    A = np.block([
        [A_s,               A_so,              B_s],
        [A_os,              A_o,               np.zeros((2, 1))],
        [np.zeros((1, 2)),  np.zeros((1, 2)),  np.array([[-lam]])],
    ])

    return A


def build_qco_measurement(
    c_qs: float,
    c_ps: float,
    c_qo: float,
    c_po: float,
) -> np.ndarray:
    """
    Measurement row:
        y = H X + v
    """
    return np.array([[c_qs, c_ps, c_qo, c_po, 0.0]], dtype=float)


def build_qco_process_covariance(
    sigma_qs: float,
    sigma_ps: float,
    sigma_qo: float,
    sigma_po: float,
    q_phi: float,
) -> np.ndarray:
    return np.diag([
        sigma_qs**2,
        sigma_ps**2,
        sigma_qo**2,
        sigma_po**2,
        q_phi,
    ]).astype(float)


def discretize_system_van_loan(A: np.ndarray, Qc: np.ndarray, dt: float):
    """
    Exact discretization of (A, Qc) using Van Loan.
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
    Qd = 0.5 * (Qd + Qd.T)

    return F, Qd