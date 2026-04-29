import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking_common import (
    SCENARIO_C_PATH,
    T5_SENSOR_ORDER,
    history_from_records,
    initialise_tracker,
    initialise_tracker_from_ais,
    load_scenario,
    make_coordinate_manager,
    nis_coverage,
    noise_covariance_from_config,
    rmse_for_window,
    truth_position_at,
)
from src.tracking_plots import plot_tracking_result


def build_t5_event_queue(measurements, active_sensors, target_id=0):
    events = []
    for measurement in measurements:
        sensor_id = measurement["sensor_id"]

        if sensor_id == "gnss":
            events.append(measurement)
            continue

        if (
            sensor_id in active_sensors
            and not measurement["is_false_alarm"]
            and measurement["target_id"] == target_id
        ):
            events.append(measurement)

    return sorted(
        events,
        key=lambda measurement: (
            measurement["time"],
            T5_SENSOR_ORDER.index(measurement["sensor_id"]),
        ),
    )


def preload_gnss_history(coord_manager, measurements):
    for measurement in measurements:
        if measurement["sensor_id"] == "gnss":
            coord_manager.update_vessel_position(
                measurement["north_m"],
                measurement["east_m"],
                measurement["time"],
            )


def t5_measurement_vector(coord_manager, measurement):
    if measurement["sensor_id"] == "ais":
        return coord_manager.compute_ais_measurement_from_report(measurement)
    return np.array([measurement["range_m"], measurement["bearing_rad"]])


def t5_noise_covariance(coord_manager, R_by_sensor, sensor_id, tracker, time_s):
    if sensor_id == "ais":
        return coord_manager.get_noise_covariance("ais", tracker.x, time_s)
    return R_by_sensor[sensor_id]


def first_initialisation_measurement(events, active_sensors):
    for measurement in events:
        if measurement["sensor_id"] in active_sensors:
            return measurement
    raise RuntimeError("No target measurement available for tracker initialization.")


def initialise_t5_tracker(first_measurement, coord_manager, R_by_sensor, sim_data):
    sensor_id = first_measurement["sensor_id"]
    if sensor_id == "ais":
        sigma_ais = sim_data["sensor_configs"]["ais"]["sigma_pos_m"]
        return initialise_tracker_from_ais(first_measurement, sigma_ais)

    return initialise_tracker(
        first_measurement,
        coord_manager,
        R_by_sensor[sensor_id],
    )


def run_t5_filter(sim_data, active_sensors, label):
    coord_manager = make_coordinate_manager(sim_data["sensor_configs"])
    preload_gnss_history(coord_manager, sim_data["measurements"])

    R_by_sensor = {}
    for sensor_id in active_sensors:
        if sensor_id in ("radar", "camera"):
            R_by_sensor[sensor_id] = noise_covariance_from_config(
                sim_data["sensor_configs"][sensor_id],
            )

    events = build_t5_event_queue(sim_data["measurements"], set(active_sensors))
    first_measurement = first_initialisation_measurement(events, set(active_sensors))
    tracker = initialise_t5_tracker(first_measurement, coord_manager, R_by_sensor, sim_data)
    last_time = first_measurement["time"]

    error_records = []
    history_records = []
    nis_records = []
    update_counts = {sensor_id: 0 for sensor_id in active_sensors}
    ais_reacquired_time = None

    for measurement in events:
        sensor_id = measurement["sensor_id"]
        time_s = measurement["time"]

        if sensor_id == "gnss":
            coord_manager.update_vessel_position(
                measurement["north_m"],
                measurement["east_m"],
                time_s,
            )
            continue

        if time_s == first_measurement["time"] and measurement is first_measurement:
            truth_pos = truth_position_at(sim_data["ground_truth"], 0, time_s)
            error = tracker.x[:2].copy() - truth_pos
            error_records.append((time_s, error))
            history_records.append((time_s, tracker.x[:2].copy(), truth_pos))
            update_counts[sensor_id] += 1
            continue

        dt = time_s - last_time
        last_time = time_s
        if dt > 0:
            tracker.predict(dt)

        z = t5_measurement_vector(coord_manager, measurement)
        h = coord_manager.compute_h(tracker.x, sensor_id, time_s)
        H = coord_manager.compute_jacobian(tracker.x, sensor_id, time_s)
        R = t5_noise_covariance(coord_manager, R_by_sensor, sensor_id, tracker, time_s)
        nis = tracker.update(z, h, H, R)

        nis_records.append((nis, 2))
        update_counts[sensor_id] += 1

        if sensor_id == "ais" and time_s > 90.0 and ais_reacquired_time is None:
            ais_reacquired_time = time_s

        truth_pos = truth_position_at(sim_data["ground_truth"], 0, time_s)
        error = tracker.x[:2].copy() - truth_pos
        error_records.append((time_s, error))
        history_records.append((time_s, tracker.x[:2].copy(), truth_pos))

    normalized_nis = np.array([nis / dof for nis, dof in nis_records])

    return {
        "label": label,
        "active_sensors": tuple(active_sensors),
        "rmse_m": rmse_for_window(error_records),
        "pre_dropout_rmse_m": rmse_for_window(error_records, None, 60.0),
        "dropout_rmse_m": rmse_for_window(error_records, 60.0, 90.0),
        "post_dropout_rmse_m": rmse_for_window(error_records, 90.0, None),
        "mean_nis_per_dof": float(np.mean(normalized_nis)) if len(normalized_nis) else None,
        "nis_95_coverage": float(nis_coverage(nis_records)) if nis_records else None,
        "num_updates": sum(update_counts.values()),
        "update_counts": update_counts,
        "ais_reacquired_time": ais_reacquired_time,
        "history": history_from_records(history_records),
    }


def format_optional_float(value, width=9, precision=3):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:{width}.{precision}f}"


def run_t5_comparison(scenario_path=SCENARIO_C_PATH, make_plot=True):
    print("Starting T5 fusion: radar + camera + AIS EKF (Scenario C)...")

    sim_data = load_scenario(scenario_path)
    without_ais = run_t5_filter(
        sim_data,
        active_sensors=("radar", "camera"),
        label="radar_camera",
    )
    with_ais = run_t5_filter(
        sim_data,
        active_sensors=("radar", "camera", "ais"),
        label="with_ais",
    )
    ais_only = run_t5_filter(
        sim_data,
        active_sensors=("ais",),
        label="ais_only",
    )

    plot_path = None
    if make_plot:
        plot_path = plot_tracking_result(
            sim_data,
            {
                "radar_camera": without_ais["history"],
                "with_ais": with_ais["history"],
                "ais_only": ais_only["history"],
            },
            "t5_ais_fusion.png",
            "T5 Radar + Camera + AIS Fusion - Scenario C",
            ais_dropout=(60.0, 90.0),
        )

    print("\nT5 validation summary")
    print("AIS dropout window: 60.0-90.0 s")
    print(
        "Run            RMSE [m]   pre-drop   dropout   post-drop   "
        "mean NIS/dof   NIS 95%   updates"
    )
    for result in (without_ais, with_ais, ais_only):
        nis_text = format_optional_float(result["mean_nis_per_dof"], width=12, precision=3)
        coverage_text = (
            "         n/a"
            if result["nis_95_coverage"] is None
            else f"{100.0 * result['nis_95_coverage']:9.1f}%"
        )
        print(
            f"{result['label']:<13}"
            f"{format_optional_float(result['rmse_m'])}"
            f"{format_optional_float(result['pre_dropout_rmse_m'])}"
            f"{format_optional_float(result['dropout_rmse_m'])}"
            f"{format_optional_float(result['post_dropout_rmse_m'])}"
            f"{nis_text}"
            f"{coverage_text}"
            f"{result['num_updates']:10d}"
        )

    print("\nSensor update counts")
    for result in (without_ais, with_ais, ais_only):
        print(f"{result['label']:<13} {result['update_counts']}")

    if with_ais["ais_reacquired_time"] is not None:
        print(f"\nAIS re-acquired at t={with_ais['ais_reacquired_time']:.1f}s after dropout.")

    improvement = without_ais["rmse_m"] - with_ais["rmse_m"]
    print(f"Overall AIS RMSE improvement: {improvement:.3f} m.")
    if plot_path is not None:
        print(f"Tracking plot: {plot_path}")

    return {
        "without_ais": without_ais,
        "with_ais": with_ais,
        "ais_only": ais_only,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    run_t5_comparison()
