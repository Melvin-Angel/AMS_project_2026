import json
from pathlib import Path

import numpy as np

from src.coordinate_manager import CoordinateFrameManager
from src.ekf_tracker import EKFTracker


BASE_DIR = Path(__file__).resolve().parents[1]
SCENARIO_A_PATH = BASE_DIR / "AMS_project_2026/harbour_sim_output/scenario_A.json"
SCENARIO_B_PATH = BASE_DIR / "AMS_project_2026/harbour_sim_output/scenario_B.json"
SCENARIO_C_PATH = BASE_DIR / "AMS_project_2026/harbour_sim_output/scenario_C.json"
TRACKING_OUTPUT_DIR = BASE_DIR / "tracking_output"

SENSOR_ORDER = ("radar", "camera")
T5_SENSOR_ORDER = ("gnss", "radar", "camera", "ais")
CHI2_95_THRESHOLDS = {
    2: 5.991,
    4: 9.488,
}


def load_scenario(path):
    with open(path, "r") as file:
        return json.load(file)


def polar_to_cartesian(range_m, bearing_rad, sensor_position=(0.0, 0.0)):
    """Convert range/bearing into an absolute NED position."""
    sensor_position = np.array(sensor_position, dtype=float)
    relative_position = np.array([
        range_m * np.cos(bearing_rad),
        range_m * np.sin(bearing_rad),
    ])
    return sensor_position + relative_position


def polar_position_covariance(range_m, bearing_rad, R_polar):
    """Transform range/bearing covariance into NED position covariance."""
    jacobian = np.array([
        [np.cos(bearing_rad), -range_m * np.sin(bearing_rad)],
        [np.sin(bearing_rad), range_m * np.cos(bearing_rad)],
    ])
    return jacobian @ R_polar @ jacobian.T


def truth_position_at(ground_truth, target_id, time_s):
    """Interpolate simulator ground truth to a requested measurement time."""
    truth = np.array(ground_truth[str(target_id)], dtype=float)
    times = truth[:, 0]
    north = np.interp(time_s, times, truth[:, 1])
    east = np.interp(time_s, times, truth[:, 2])
    return np.array([north, east])


def real_sensor_measurements(measurements, sensor_ids, target_id=0):
    return [
        measurement for measurement in measurements
        if (
            measurement["sensor_id"] in sensor_ids
            and not measurement["is_false_alarm"]
            and measurement["target_id"] == target_id
        )
    ]


def real_radar_measurements(measurements, target_id=0):
    return real_sensor_measurements(measurements, {"radar"}, target_id)


def noise_covariance_from_config(sensor_config):
    sigma_r = sensor_config["sigma_r_m"]
    sigma_phi = np.deg2rad(sensor_config["sigma_phi_deg"])
    return np.diag([sigma_r**2, sigma_phi**2])


def make_coordinate_manager(sensor_configs):
    camera_pos = sensor_configs.get("camera", {}).get("pos_ned", (-80.0, 120.0))
    ais_sigma = sensor_configs.get("ais", {}).get("sigma_pos_m", 4.0)
    return CoordinateFrameManager(camera_pos=camera_pos, sigma_ais=ais_sigma)


def initialise_tracker(first_measurement, coord_manager, R_measurement):
    sensor_position = coord_manager.get_sensor_position(first_measurement["sensor_id"])
    z0 = np.array([first_measurement["range_m"], first_measurement["bearing_rad"]])
    position = polar_to_cartesian(z0[0], z0[1], sensor_position)
    position_cov = polar_position_covariance(z0[0], z0[1], R_measurement)

    x_init = np.array([position[0], position[1], 0.0, 0.0])
    P_init = np.zeros((4, 4))
    P_init[:2, :2] = position_cov
    P_init[2:, 2:] = np.eye(2) * 25.0

    return EKFTracker(x_initial=x_init, P_initial=P_init, sigma_a=0.05)


def initialise_tracker_from_ais(first_measurement, sigma_ais):
    x_init = np.array([
        first_measurement["north_m"],
        first_measurement["east_m"],
        0.0,
        0.0,
    ])
    P_init = np.zeros((4, 4))
    P_init[:2, :2] = np.eye(2) * sigma_ais**2
    P_init[2:, 2:] = np.eye(2) * 25.0
    return EKFTracker(x_initial=x_init, P_initial=P_init, sigma_a=0.05)


def group_measurements_by_time(measurements):
    grouped = {}
    for measurement in measurements:
        grouped.setdefault(measurement["time"], []).append(measurement)
    return grouped


def ordered_sensor_measurements(measurements, sensor_order=SENSOR_ORDER):
    return sorted(
        measurements,
        key=lambda measurement: sensor_order.index(measurement["sensor_id"]),
    )


def block_diag(matrices):
    rows = sum(matrix.shape[0] for matrix in matrices)
    cols = sum(matrix.shape[1] for matrix in matrices)
    result = np.zeros((rows, cols))
    row = 0
    col = 0
    for matrix in matrices:
        n_rows, n_cols = matrix.shape
        result[row:row + n_rows, col:col + n_cols] = matrix
        row += n_rows
        col += n_cols
    return result


def nis_coverage(nis_records):
    covered = 0
    for nis, dof in nis_records:
        threshold = CHI2_95_THRESHOLDS[dof]
        covered += nis <= threshold
    return covered / len(nis_records)


def rmse_for_window(error_records, start_s=None, end_s=None):
    selected = []
    for time_s, error_vector in error_records:
        if start_s is not None and time_s < start_s:
            continue
        if end_s is not None and time_s > end_s:
            continue
        selected.append(error_vector)

    if not selected:
        return None

    selected = np.array(selected)
    return float(np.sqrt(np.mean(np.sum(selected**2, axis=1))))


def history_from_records(records):
    if not records:
        return {
            "times": np.array([]),
            "estimates": np.empty((0, 2)),
            "truth": np.empty((0, 2)),
        }

    times = np.array([record[0] for record in records])
    estimates = np.array([record[1] for record in records])
    truth = np.array([record[2] for record in records])
    return {
        "times": times,
        "estimates": estimates,
        "truth": truth,
    }
