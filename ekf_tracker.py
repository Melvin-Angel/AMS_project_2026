import numpy as np


def wrap_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class EKFTracker:
    def __init__(self, x_inicial, P_inicial, sigma_a):
        """
        Inicializa a memória do Filtro de Kalman para UM alvo.
        """
        # Estado atual [Norte, Este, Vel_Norte, Vel_Este]
        self.x = x_inicial
        
        # Matriz de covariância (Incerteza atual)
        self.P = P_inicial
        
        # Desvio padrão da aceleração (O ruído de processo Q)
        # (O valor de sigma_a costuma estar especificado no enunciado, ex: 0.05)
        self.sigma_a = sigma_a
        self.last_nis = None

    def predict(self, dt):
        """
        Fase 1 do EKF: Prevê onde o barco vai estar com base na velocidade.
        """
        # 1. Matriz de Transição de Estado (F) - Modelo CV
        # Assume que a posição muda com dt*velocidade, e a velocidade mantém-se constante.
        F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # 2. Matriz de Ruído de Processo (Q)
        # Modela as pequenas variações de aceleração que não conseguimos prever
        q11 = (dt**4) / 4.0
        q13 = (dt**3) / 2.0
        q33 = dt**2

        Q = (self.sigma_a**2) * np.array([
            [q11, 0.0, q13, 0.0],
            [0.0, q11, 0.0, q13],
            [q13, 0.0, q33, 0.0],
            [0.0, q13, 0.0, q33]
        ])

        # 3. Equações Matemáticas de Predição do Kalman
        self.x = F @ self.x                    # Estado previsto
        self.P = F @ self.P @ F.T + Q          # Incerteza prevista

        return self.x

    def update(self, z, h, H, R):
        """
        Fase 2 do EKF: corrige a previsão usando uma medição range/bearing.

        Returns
        -------
        float
            NIS = innovation.T S^-1 innovation, used for consistency checks.
        """
        innovation = z - h
        innovation[1] = wrap_angle(innovation[1])

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T

        self.x = self.x + K @ innovation

        I = np.eye(self.P.shape[0])
        # Joseph form keeps P symmetric positive semi-definite under rounding.
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.last_nis = float(innovation.T @ np.linalg.solve(S, innovation))
        return self.last_nis
