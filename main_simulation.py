import json
from pathlib import Path

import numpy as np

from coordinate_manager import CoordinateFrameManager
from ekf_tracker import EKFTracker


BASE_DIR = Path(__file__).resolve().parent
SCENARIO_A_PATH = BASE_DIR / "AMS_project_2026/harbour_sim_output/scenario_A.json"


def polar_to_cartesian(range_m, bearing_rad):
    """Convert radar range/bearing into NED position."""
    return np.array([
        range_m * np.cos(bearing_rad),
        range_m * np.sin(bearing_rad),
    ])


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


def real_radar_measurements(measurements, target_id=0):
    return [
        m for m in measurements
        if (
            m["sensor_id"] == "radar"
            and not m["is_false_alarm"]
            and m["target_id"] == target_id
        )
    ]


def initialise_tracker(first_measurement, R_radar):
    z0 = np.array([first_measurement["range_m"], first_measurement["bearing_rad"]])
    position = polar_to_cartesian(z0[0], z0[1])
    position_cov = polar_position_covariance(z0[0], z0[1], R_radar)

    x_init = np.array([position[0], position[1], 0.0, 0.0])
    P_init = np.zeros((4, 4))
    P_init[:2, :2] = position_cov
    P_init[2:, 2:] = np.eye(2) * 25.0

    return EKFTracker(x_inicial=x_init, P_inicial=P_init, sigma_a=0.05)


def run_simulation(scenario_path=SCENARIO_A_PATH):
    print("A iniciar o rastreamento T3: EKF com radar apenas (Cenario A)...")

    coord_manager = CoordinateFrameManager()

    with open(scenario_path, "r") as file:
        sim_data = json.load(file)

    sigma_r = sim_data["sensor_configs"]["radar"]["sigma_r_m"]
    sigma_phi = np.deg2rad(sim_data["sensor_configs"]["radar"]["sigma_phi_deg"])
    R_radar = np.diag([sigma_r**2, sigma_phi**2])

    radar_measurements = real_radar_measurements(sim_data["measurements"], target_id=0)
    if not radar_measurements:
        raise RuntimeError("No real radar detections found for Scenario A.")

    tracker = initialise_tracker(radar_measurements[0], R_radar)
    last_time = radar_measurements[0]["time"]

    estimates = []
    truth_positions = []
    nis_values = []

    update_measurements = radar_measurements[1:]

    for k, measurement in enumerate(update_measurements):
        time_s = measurement["time"]
        dt = time_s - last_time
        last_time = time_s

        if dt > 0:
            tracker.predict(dt)

        z = np.array([measurement["range_m"], measurement["bearing_rad"]])
        h = coord_manager.compute_h(tracker.x, "radar")
        H = coord_manager.compute_jacobian(tracker.x, "radar")
        nis = tracker.update(z, h, H, R_radar)

        truth_pos = truth_position_at(sim_data["ground_truth"], target_id=0, time_s=time_s)
        estimates.append(tracker.x[:2].copy())
        truth_positions.append(truth_pos)
        nis_values.append(nis)

        if k == 0 or (k + 1) % 10 == 0:
            error_m = np.linalg.norm(tracker.x[:2] - truth_pos)
            print(
                f"[t={time_s:6.2f}s] "
                f"est=({tracker.x[0]:7.1f}, {tracker.x[1]:7.1f}) m "
                f"truth=({truth_pos[0]:7.1f}, {truth_pos[1]:7.1f}) m "
                f"err={error_m:5.1f} m NIS={nis:5.2f}"
            )

    estimates = np.array(estimates)
    truth_positions = np.array(truth_positions)
    nis_values = np.array(nis_values)

    position_errors = estimates - truth_positions
    rmse = np.sqrt(np.mean(np.sum(position_errors**2, axis=1)))
    mean_nis = float(np.mean(nis_values))
    frac_nis_below_95 = float(np.mean(nis_values <= 5.991))

    print("\nT3 validation summary")
    print(f"Radar target detections used : {len(radar_measurements)}")
    print(f"EKF correction updates       : {len(update_measurements)}")
    print(f"Position RMSE               : {rmse:.2f} m")
    print(f"Mean NIS                    : {mean_nis:.2f} (expected near 2.0)")
    print(f"NIS <= chi2_2 95% threshold : {100.0 * frac_nis_below_95:.1f}%")

    return {
        "rmse_m": rmse,
        "mean_nis": mean_nis,
        "frac_nis_below_95": frac_nis_below_95,
        "num_updates": len(update_measurements),
    }


if __name__ == "__main__":
    run_simulation()
