import numpy as np

class CoordinateFrameManager:
    def __init__(self):
        """
        Initialize the coordinate manager with sensor offsets and noise parameters.
        The origin (0,0) is fixed at the mm-wave radar position.
        """
        # Fixed land-based sensor positions in NED frame
        self.radar_pos = np.array([0.0, 0.0]) # Radar is the origin [cite: 119]
        self.camera_pos = np.array([-80.0, 120.0]) # Camera fixed offset 
        
        # Vessel position (updated via GNSS at 1Hz)
        self.vessel_pos = np.array([0.0, 0.0]) # Time-varying offset for AIS [cite: 30, 120]

        # Standard deviations from project specifications [cite: 188, 189]
        self.sigma_r_radar = 5.0
        self.sigma_phi_radar = np.radians(0.3)
        
        self.sigma_r_camera = 8.0
        self.sigma_phi_camera = np.radians(0.15)
        
        # AIS position noise (used for implied range/bearing)
        self.sigma_ais = 4.0

    def update_vessel_position(self, north_m, east_m):
        """Update the current vessel NED position from GNSS fix."""
        self.vessel_pos = np.array([north_m, east_m])

    def get_sensor_position(self, sensor_id):
        """Returns the NED position of the specified sensor."""
        if sensor_id == 'radar':
            return self.radar_pos
        elif sensor_id == 'camera':
            return self.camera_pos
        elif sensor_id == 'ais':
            return self.vessel_pos # AIS moves with the vessel [cite: 36, 123]
        else:
            raise ValueError(f"Unknown sensor: {sensor_id}")

    def compute_h(self, state_x, sensor_id):
        """
        Computes the measurement function h_i(x, t).
        Maps the global state x to the sensor-relative range and bearing.
        """
        p_N, p_E = state_x[0], state_x[1]
        s_i = self.get_sensor_position(sensor_id)
        
        dn = p_N - s_i[0]
        de = p_E - s_i[1]
        
        # Range calculation
        r = np.sqrt(dn**2 + de**2)
        # Bearing calculation (atan2(East, North) for clockwise from North) [cite: 59, 60]
        phi = np.arctan2(de, dn)
        
        return np.array([r, phi])

    def compute_jacobian(self, state_x, sensor_id):
        """
        Computes the Jacobian matrix H_i for the measurement function.
        H is a 2x4 matrix: [d_range/dx, d_bearing/dx].
        """
        p_N, p_E = state_x[0], state_x[1]
        s_i = self.get_sensor_position(sensor_id)
        
        dn = p_N - s_i[0]
        de = p_E - s_i[1]
        
        r2 = dn**2 + de**2
        
        if r2 < 1e-12:
            r2 = 1e-12 # Avoid singularity at the sensor origin

        r = np.sqrt(r2)
        
        H = np.zeros((2, 4))
        
        # Partial derivatives for Range
        H[0, 0] = dn / r
        H[0, 1] = de / r
        
        # Partial derivatives for Bearing
        H[1, 0] = -de / r2
        H[1, 1] = dn / r2
        
        return H

    def get_noise_covariance(self, sensor_id):
        """Returns the 2x2 measurement noise covariance matrix R_i."""
        if sensor_id == 'radar':
            return np.diag([self.sigma_r_radar**2, self.sigma_phi_radar**2])
        elif sensor_id == 'camera':
            return np.diag([self.sigma_r_camera**2, self.sigma_phi_camera**2])
        elif sensor_id == 'ais':
            # Simplified range/bearing noise for AIS position uncertainty
            # In a real scenario, this might be transformed from Cartesian noise
            return np.diag([self.sigma_ais**2, 1e-4]) 
        else:
            return np.eye(2)
    def convert_ais_to_polar(self, target_N, target_E):
        """
        Converte a posição absoluta (Norte, Este) recebida pelo AIS 
        para coordenadas polares (Range, Bearing) relativas ao nosso navio.
        """
        # Distância entre o alvo e o nosso navio
        delta_n = target_N - self.vessel_pos[0]
        delta_e = target_E - self.vessel_pos[1]
        
        # Pitágoras para a distância (range) e atan2 para o ângulo (bearing)
        r = np.sqrt(delta_n**2 + delta_e**2)
        phi = np.arctan2(delta_e, delta_n)
        
        return np.array([r, phi])
