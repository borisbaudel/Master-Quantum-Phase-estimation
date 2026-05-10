import numpy as np

def observability_matrix(A, C):
    n = A.shape[0]
    O = C
    M = C

    for _ in range(1, n):
        M = M @ A
        O = np.vstack((O, M))

    return O


def observability_spectrum(A, C):
    O = observability_matrix(A, C)

    U, S, Vh = np.linalg.svd(O)

    return S


def lambda_min_obsv(A, C):
    S = observability_spectrum(A, C)
    return S[-1]  # plus petite singular value


def cond_obsv(A, C):
    S = observability_spectrum(A, C)
    return S[0] / S[-1]

def fisher_information(theta, R=1.0):
    H = np.array([[np.cos(theta), np.sin(theta), 0.0]])
    F = H.T @ H / R
    return F


def fisher_min_eig(theta):
    F = fisher_information(theta)
    eigvals = np.linalg.eigvalsh(F)
    return np.min(eigvals)