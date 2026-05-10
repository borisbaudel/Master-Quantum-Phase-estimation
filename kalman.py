import numpy as np


class DiscreteKalmanFilter:
    """
    Standard discrete-time Kalman filter:

        x_{k+1} = F x_k + w_k
        y_k     = H x_k + v_k

    with:
        E[w_k w_k^T] = Q
        E[v_k v_k^T] = R
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Qd: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> None:
        self.F = F.astype(float).copy()
        self.H = H.astype(float).copy()
        self.Q = Qd.astype(float).copy()
        self.R = R.astype(float).copy()

        n = self.F.shape[0]
        self.x = np.zeros(n, dtype=float) if x0 is None else x0.astype(float).copy()
        self.P = np.eye(n, dtype=float) if P0 is None else P0.astype(float).copy()

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y: np.ndarray) -> None:
        y = np.atleast_1d(y).astype(float)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        innovation = y - self.H @ self.x
        self.x = self.x + K @ innovation

        I = np.eye(self.P.shape[0], dtype=float)
        self.P = (I - K @ self.H) @ self.P

        # Symmetrize covariance to reduce numerical drift
        self.P = 0.5 * (self.P + self.P.T)

    def step(self, y: np.ndarray) -> np.ndarray:
        self.predict()
        self.update(y)
        return self.x.copy()