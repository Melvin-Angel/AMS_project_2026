import numpy as np


def wrap_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class EKFTracker:
    def __init__(self, x_initial, P_initial, sigma_a):
        """
        Initialize the Kalman filter state for one target.
        """
        # Current state [North, East, Velocity North, Velocity East]
        self.x = x_initial
        
        # Current covariance matrix
        self.P = P_initial
        
        # Acceleration standard deviation used to build the process noise Q.
        self.sigma_a = sigma_a
        self.last_nis = None

    def predict(self, dt):
        """
        EKF phase 1: predict the target state using the current velocity.
        """
        # 1. State transition matrix (F), using a constant-velocity model.
        # Position changes by dt * velocity, while velocity remains constant.
        F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # 2. Process noise matrix (Q).
        # Models small acceleration changes that the motion model cannot predict.
        q11 = (dt**4) / 4.0
        q13 = (dt**3) / 2.0
        q33 = dt**2

        Q = (self.sigma_a**2) * np.array([
            [q11, 0.0, q13, 0.0],
            [0.0, q11, 0.0, q13],
            [q13, 0.0, q33, 0.0],
            [0.0, q13, 0.0, q33]
        ])

        # 3. Kalman prediction equations.
        self.x = F @ self.x                    # Predicted state
        self.P = F @ self.P @ F.T + Q          # Predicted covariance

        return self.x

    def update(self, z, h, H, R, angle_indices=(1,)):
        """
        EKF phase 2: correct the prediction using one or more measurements.

        Returns
        -------
        float
            NIS = innovation.T S^-1 innovation, used for consistency checks.
        """
        innovation = z - h
        for angle_index in angle_indices:
            innovation[angle_index] = wrap_angle(innovation[angle_index])

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T

        self.x = self.x + K @ innovation

        I = np.eye(self.P.shape[0])
        # Joseph form keeps P symmetric positive semi-definite under rounding.
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.last_nis = float(innovation.T @ np.linalg.solve(S, innovation))
        return self.last_nis
