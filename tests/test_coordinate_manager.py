import unittest

import numpy as np

from coordinate_manager import CoordinateFrameManager


class TestCoordinateFrameManager(unittest.TestCase):
    def setUp(self):
        self.manager = CoordinateFrameManager()

    def test_known_target_generates_expected_radar_and_camera_measurements(self):
        state = np.array([300.0, 400.0, 2.0, -1.0])

        radar_expected = np.array([500.0, np.arctan2(400.0, 300.0)])
        np.testing.assert_allclose(
            self.manager.compute_h(state, "radar"),
            radar_expected,
            atol=1e-12,
        )

        camera_delta_n = 300.0 - (-80.0)
        camera_delta_e = 400.0 - 120.0
        camera_expected = np.array([
            np.hypot(camera_delta_n, camera_delta_e),
            np.arctan2(camera_delta_e, camera_delta_n),
        ])
        np.testing.assert_allclose(
            self.manager.compute_h(state, "camera"),
            camera_expected,
            atol=1e-12,
        )

    def test_measurement_jacobian_matches_known_input(self):
        state = np.array([300.0, 400.0, 0.0, 0.0])

        expected_H = np.array([
            [0.6, 0.8, 0.0, 0.0],
            [-0.0016, 0.0012, 0.0, 0.0],
        ])

        np.testing.assert_allclose(
            self.manager.compute_jacobian(state, "radar"),
            expected_H,
            atol=1e-12,
        )

    def test_ais_uses_closest_gnss_fix_for_position_to_observation_conversion(self):
        self.manager.update_vessel_position(10.0, 20.0, time_s=0.0)
        self.manager.update_vessel_position(100.0, 200.0, time_s=10.0)

        expected = np.array([50.0, np.arctan2(40.0, 30.0)])

        np.testing.assert_allclose(
            self.manager.convert_ais_to_polar(130.0, 240.0, time_s=8.0),
            expected,
            atol=1e-12,
        )

        state = np.array([130.0, 240.0, 0.0, 0.0])
        np.testing.assert_allclose(
            self.manager.compute_h(state, "ais", time_s=8.0),
            expected,
            atol=1e-12,
        )

    def test_ais_conversion_is_consistent_with_radar_at_same_location(self):
        self.manager.update_vessel_position(0.0, 0.0, time_s=5.0)
        state = np.array([300.0, 400.0, 0.0, 0.0])

        radar_measurement = self.manager.compute_h(state, "radar")
        ais_measurement = self.manager.convert_ais_to_polar(300.0, 400.0, time_s=5.0)

        np.testing.assert_allclose(ais_measurement, radar_measurement, atol=1e-12)

    def test_noise_covariances_match_sensor_specs(self):
        np.testing.assert_allclose(
            self.manager.get_noise_covariance("radar"),
            np.diag([25.0, np.deg2rad(0.3) ** 2]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.manager.get_noise_covariance("camera"),
            np.diag([64.0, np.deg2rad(0.15) ** 2]),
            atol=1e-12,
        )

    def test_ais_cartesian_noise_is_transformed_to_polar_covariance(self):
        self.manager.update_vessel_position(100.0, 200.0, time_s=10.0)
        state = np.array([130.0, 240.0, 0.0, 0.0])

        np.testing.assert_allclose(
            self.manager.get_noise_covariance("ais", state, time_s=10.0),
            np.diag([16.0, 0.0064]),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
