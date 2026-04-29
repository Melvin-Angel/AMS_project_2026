import numpy as np

from tracking_common import (
    SCENARIO_B_PATH,
    SENSOR_ORDER,
    block_diag,
    group_measurements_by_time,
    history_from_records,
    initialise_tracker,
    load_scenario,
    nis_coverage,
    noise_covariance_from_config,
    ordered_sensor_measurements,
    real_sensor_measurements,
    truth_position_at,
)
from tracking_plots import plot_tracking_result
from coordinate_manager import CoordinateFrameManager


def build_joint_measurement(tracker, coord_manager, measurements, R_by_sensor):
    z_parts = []
    h_parts = []
    H_parts = []
    R_parts = []
    angle_indices = []
    offset = 0

    for measurement in ordered_sensor_measurements(measurements):
        sensor_id = measurement["sensor_id"]
        z_parts.append(np.array([measurement["range_m"], measurement["bearing_rad"]]))
        h_parts.append(coord_manager.compute_h(tracker.x, sensor_id))
        H_parts.append(coord_manager.compute_jacobian(tracker.x, sensor_id))
        R_parts.append(R_by_sensor[sensor_id])
        angle_indices.append(offset + 1)
        offset += 2

    return (
        np.concatenate(z_parts),
        np.concatenate(h_parts),
        np.vstack(H_parts),
        block_diag(R_parts),
        tuple(angle_indices),
    )


def run_t4_filter(sim_data, mode, sensor_ids=SENSOR_ORDER, label=None):
    coord_manager = CoordinateFrameManager()
    R_by_sensor = {
        sensor_id: noise_covariance_from_config(sim_data["sensor_configs"][sensor_id])
        for sensor_id in sensor_ids
    }

    measurements = real_sensor_measurements(
        sim_data["measurements"],
        set(sensor_ids),
        target_id=0,
    )
    if not measurements:
        raise RuntimeError("No real radar/camera detections found for Scenario B.")

    measurements_by_time = group_measurements_by_time(measurements)
    scan_times = sorted(measurements_by_time)
    first_measurement = ordered_sensor_measurements(measurements_by_time[scan_times[0]])[0]
    tracker = initialise_tracker(
        first_measurement,
        coord_manager,
        R_by_sensor[first_measurement["sensor_id"]],
    )
    last_time = first_measurement["time"]

    history_records = []
    nis_records = []
    joint_updates = 0
    scalar_sensor_updates = 0

    for time_s in scan_times[1:]:
        dt = time_s - last_time
        last_time = time_s
        if dt > 0:
            tracker.predict(dt)

        scan_measurements = ordered_sensor_measurements(measurements_by_time[time_s])

        if mode == "sequential":
            for measurement in scan_measurements:
                sensor_id = measurement["sensor_id"]
                z = np.array([measurement["range_m"], measurement["bearing_rad"]])
                h = coord_manager.compute_h(tracker.x, sensor_id)
                H = coord_manager.compute_jacobian(tracker.x, sensor_id)
                nis = tracker.update(z, h, H, R_by_sensor[sensor_id])
                nis_records.append((nis, 2))
                scalar_sensor_updates += 1
        elif mode == "joint":
            z, h, H, R, angle_indices = build_joint_measurement(
                tracker,
                coord_manager,
                scan_measurements,
                R_by_sensor,
            )
            nis = tracker.update(z, h, H, R, angle_indices=angle_indices)
            nis_records.append((nis, len(z)))
            joint_updates += 1
        else:
            raise ValueError(f"Unknown T4 mode: {mode}")

        truth_pos = truth_position_at(sim_data["ground_truth"], target_id=0, time_s=time_s)
        history_records.append((time_s, tracker.x[:2].copy(), truth_pos))

    history = history_from_records(history_records)
    position_errors = history["estimates"] - history["truth"]
    rmse = np.sqrt(np.mean(np.sum(position_errors**2, axis=1)))
    normalized_nis_values = np.array([nis / dof for nis, dof in nis_records])

    return {
        "mode": label or mode,
        "rmse_m": float(rmse),
        "mean_nis_per_dof": float(np.mean(normalized_nis_values)),
        "nis_95_coverage": float(nis_coverage(nis_records)),
        "num_scans": len(scan_times),
        "num_nis_updates": len(nis_records),
        "num_joint_updates": joint_updates,
        "num_sensor_updates": scalar_sensor_updates,
        "history": history,
    }


def run_t4_comparison(scenario_path=SCENARIO_B_PATH, make_plot=True):
    print("Starting T4 fusion: radar + stereo camera EKF (Scenario B)...")

    sim_data = load_scenario(scenario_path)
    radar_only = run_t4_filter(
        sim_data,
        mode="sequential",
        sensor_ids=("radar",),
        label="radar_only",
    )
    sequential = run_t4_filter(sim_data, mode="sequential")
    joint = run_t4_filter(sim_data, mode="joint")

    measurements = real_sensor_measurements(
        sim_data["measurements"],
        set(SENSOR_ORDER),
        target_id=0,
    )
    measurements_by_time = group_measurements_by_time(measurements)
    simultaneous_scans = sum(
        {measurement["sensor_id"] for measurement in scan} == set(SENSOR_ORDER)
        for scan in measurements_by_time.values()
    )

    plot_path = None
    if make_plot:
        plot_path = plot_tracking_result(
            sim_data,
            {
                "radar_only": radar_only["history"],
                "sequential": sequential["history"],
                "joint": joint["history"],
            },
            "t4_camera_fusion.png",
            "T4 Radar + Camera Fusion - Scenario B",
        )

    print("\nT4 validation summary")
    print(f"Real radar/camera scans       : {len(measurements_by_time)}")
    print(f"Simultaneous radar+camera scans: {simultaneous_scans}")
    print("Architecture   RMSE [m]    mean NIS/dof   NIS 95% coverage   updates")
    for result in (radar_only, sequential, joint):
        print(
            f"{result['mode']:<12} "
            f"{result['rmse_m']:9.3f} "
            f"{result['mean_nis_per_dof']:14.3f} "
            f"{100.0 * result['nis_95_coverage']:16.1f}% "
            f"{result['num_nis_updates']:9d}"
        )

    better_rmse = min((sequential, joint), key=lambda result: result["rmse_m"])
    better_nis = min(
        (sequential, joint),
        key=lambda result: abs(result["mean_nis_per_dof"] - 1.0),
    )
    print(
        f"\nLower RMSE: {better_rmse['mode']} "
        f"({better_rmse['rmse_m']:.2f} m)."
    )
    print(
        f"Better mean NIS consistency: {better_nis['mode']} "
        f"(mean NIS/dof {better_nis['mean_nis_per_dof']:.2f}; ideal is 1.0)."
    )
    if plot_path is not None:
        print(f"Tracking plot: {plot_path}")

    return {
        "radar_only": radar_only,
        "sequential": sequential,
        "joint": joint,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    run_t4_comparison()
