import numpy as np


class CoordinateFrameManager:
    """
    Measurement-model manager for the shared NED coordinate frame.

    State convention:
        x = [north_m, east_m, velocity_north_mps, velocity_east_mps]

    Measurement convention:
        z = [range_m, bearing_rad], with bearing measured clockwise from North
        as atan2(East, North).
    """

    def __init__(
        self,
        radar_pos=(0.0, 0.0),
        camera_pos=(-80.0, 120.0),
        sigma_r_radar=5.0,
        sigma_phi_radar_deg=0.3,
        sigma_r_camera=8.0,
        sigma_phi_camera_deg=0.15,
        sigma_ais=4.0,
    ):
        """
        Initialize sensor offsets and noise parameters from the project specs.

        The NED origin is fixed at the mm-wave radar, so the radar offset is
        normally [0, 0]. The AIS receiver is mounted on the moving vessel; its
        effective sensor offset is the closest stored GNSS position.
        """
        self.radar_pos = np.array(radar_pos, dtype=float)
        self.camera_pos = np.array(camera_pos, dtype=float)

        self.vessel_pos = np.array([0.0, 0.0], dtype=float)
        self._gnss_history = []

        self.sigma_r_radar = float(sigma_r_radar)
        self.sigma_phi_radar = np.deg2rad(sigma_phi_radar_deg)

        self.sigma_r_camera = float(sigma_r_camera)
        self.sigma_phi_camera = np.deg2rad(sigma_phi_camera_deg)

        self.sigma_ais = float(sigma_ais)

    def update_vessel_position(self, north_m, east_m, time_s=None):
        """
        Store a GNSS fix for the own vessel.

        If a timestamp is provided, the fix is retained so AIS updates can use
        the vessel position closest in time to the AIS report. Without a
        timestamp, this method still updates the latest vessel position for
        backwards compatibility with the T3 code.
        """
        position = np.array([north_m, east_m], dtype=float)
        self.vessel_pos = position

        if time_s is not None:
            self._gnss_history.append((float(time_s), position))
            self._gnss_history.sort(key=lambda item: item[0])

    def get_sensor_position(self, sensor_id, time_s=None):
        """Return the NED position of the requested sensor."""
        if sensor_id == "radar":
            return self.radar_pos
        if sensor_id == "camera":
            return self.camera_pos
        if sensor_id == "ais":
            return self.get_vessel_position(time_s)
        raise ValueError(f"Unknown sensor: {sensor_id}")

    def get_vessel_position(self, time_s=None):
        """
        Return the latest vessel position, or the stored GNSS fix closest to
        the requested timestamp.
        """
        if time_s is None or not self._gnss_history:
            return self.vessel_pos

        query_time = float(time_s)
        _, closest_position = min(
            self._gnss_history,
            key=lambda item: abs(item[0] - query_time),
        )
        return closest_position

    def compute_h(self, state_x, sensor_id, time_s=None):
        """
        Compute the measurement function h_i(x, t) for radar, camera, or AIS.

        Radar and camera use fixed NED offsets. AIS uses the vessel GNSS
        position as a time-varying offset and returns the implied range/bearing
        from the vessel to the target state.
        """
        p_north, p_east = state_x[0], state_x[1]
        sensor_position = self.get_sensor_position(sensor_id, time_s)
        return self._relative_position_to_polar(
            p_north - sensor_position[0],
            p_east - sensor_position[1],
        )

    def compute_jacobian(self, state_x, sensor_id, time_s=None):
        """
        Compute the 2x4 Jacobian H_i for the measurement function.

        Only target position appears in range/bearing measurements, so the
        velocity columns are zero.
        """
        p_north, p_east = state_x[0], state_x[1]
        sensor_position = self.get_sensor_position(sensor_id, time_s)

        delta_n = p_north - sensor_position[0]
        delta_e = p_east - sensor_position[1]

        r2 = delta_n**2 + delta_e**2
        if r2 < 1e-12:
            r2 = 1e-12

        r = np.sqrt(r2)

        H = np.zeros((2, 4))
        H[0, 0] = delta_n / r
        H[0, 1] = delta_e / r
        H[1, 0] = -delta_e / r2
        H[1, 1] = delta_n / r2

        return H

    def get_noise_covariance(self, sensor_id, state_x=None, time_s=None):
        """
        Return the 2x2 measurement noise covariance R_i.

        Radar and camera are specified directly in range/bearing units. AIS is
        specified as Cartesian NED position noise, so when a state is supplied
        it is transformed into the implied range/bearing covariance.
        """
        if sensor_id == "radar":
            return np.diag([self.sigma_r_radar**2, self.sigma_phi_radar**2])

        if sensor_id == "camera":
            return np.diag([self.sigma_r_camera**2, self.sigma_phi_camera**2])

        if sensor_id == "ais":
            if state_x is None:
                return np.diag([self.sigma_ais**2, self.sigma_ais**2])
            return self._cartesian_position_noise_to_polar_covariance(
                state_x,
                "ais",
                time_s,
            )

        raise ValueError(f"Unknown sensor: {sensor_id}")

    def convert_ais_to_polar(self, target_north, target_east, time_s=None):
        """
        Convert an absolute AIS target position into range/bearing relative to
        the own vessel GNSS position.
        """
        vessel_position = self.get_vessel_position(time_s)
        return self._relative_position_to_polar(
            target_north - vessel_position[0],
            target_east - vessel_position[1],
        )

    def compute_ais_measurement_from_report(self, ais_report, time_s=None):
        """
        Convert an AIS report dict with north_m/east_m fields to range/bearing.
        """
        report_time = ais_report.get("time", time_s)
        return self.convert_ais_to_polar(
            ais_report["north_m"],
            ais_report["east_m"],
            report_time,
        )

    def _cartesian_position_noise_to_polar_covariance(
        self,
        state_x,
        sensor_id,
        time_s=None,
    ):
        """Transform isotropic Cartesian position noise into polar covariance."""
        H_pos = self.compute_jacobian(state_x, sensor_id, time_s)[:, :2]
        R_cartesian = np.eye(2) * self.sigma_ais**2
        return H_pos @ R_cartesian @ H_pos.T

    @staticmethod
    def _relative_position_to_polar(delta_north, delta_east):
        """Convert a relative NED position to range and bearing."""
        range_m = np.hypot(delta_north, delta_east)
        bearing_rad = np.arctan2(delta_east, delta_north)
        return np.array([range_m, bearing_rad])
