import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coordinate_manager import CoordinateFrameManager
from src.tracking_common import (
    SCENARIO_A_PATH,
    history_from_records,
    initialise_tracker,
    load_scenario,
    noise_covariance_from_config,
    real_radar_measurements,
    truth_position_at,
)
from src.tracking_plots import plot_tracking_result


def run_t3_radar_tracker(scenario_path=SCENARIO_A_PATH, make_plot=True):
    print("Starting T3 tracking: radar-only EKF (Scenario A)...")

    coord_manager = CoordinateFrameManager()
    sim_data = load_scenario(scenario_path)
    R_radar = noise_covariance_from_config(sim_data["sensor_configs"]["radar"])

    radar_measurements = real_radar_measurements(sim_data["measurements"], target_id=0)
    if not radar_measurements:
        raise RuntimeError("No real radar detections found for Scenario A.")

    tracker = initialise_tracker(radar_measurements[0], coord_manager, R_radar)
    last_time = radar_measurements[0]["time"]

    history_records = []
    nis_values = []

    for k, measurement in enumerate(radar_measurements[1:]):
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
        history_records.append((time_s, tracker.x[:2].copy(), truth_pos))
        nis_values.append(nis)

        if k == 0 or (k + 1) % 10 == 0:
            error_m = np.linalg.norm(tracker.x[:2] - truth_pos)
            print(
                f"[t={time_s:6.2f}s] "
                f"est=({tracker.x[0]:7.1f}, {tracker.x[1]:7.1f}) m "
                f"truth=({truth_pos[0]:7.1f}, {truth_pos[1]:7.1f}) m "
                f"err={error_m:5.1f} m NIS={nis:5.2f}"
            )

    history = history_from_records(history_records)
    position_errors = history["estimates"] - history["truth"]
    rmse = np.sqrt(np.mean(np.sum(position_errors**2, axis=1)))
    nis_values = np.array(nis_values)
    mean_nis = float(np.mean(nis_values))
    frac_nis_below_95 = float(np.mean(nis_values <= 5.991))

    plot_path = None
    if make_plot:
        plot_path = plot_tracking_result(
            sim_data,
            {"radar_only": history},
            "t3_radar_tracker.png",
            "T3 Radar-only EKF - Scenario A",
        )

    print("\nT3 validation summary")
    print(f"Radar target detections used : {len(radar_measurements)}")
    print(f"EKF correction updates       : {len(radar_measurements) - 1}")
    print(f"Position RMSE               : {rmse:.2f} m")
    print(f"Mean NIS                    : {mean_nis:.2f} (expected near 2.0)")
    print(f"NIS <= chi2_2 95% threshold : {100.0 * frac_nis_below_95:.1f}%")
    if plot_path is not None:
        print(f"Tracking plot               : {plot_path}")

    return {
        "rmse_m": float(rmse),
        "mean_nis": mean_nis,
        "frac_nis_below_95": frac_nis_below_95,
        "num_updates": len(radar_measurements) - 1,
        "history": history,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    run_t3_radar_tracker()
