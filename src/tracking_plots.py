import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

from src.tracking_common import TRACKING_OUTPUT_DIR, polar_to_cartesian


SENSOR_COLORS = {
    "radar": "#2E75B6",
    "camera": "#0B6E4F",
    "ais": "#BA7517",
    "gnss": "#7F77DD",
}
RUN_COLORS = ["#111827", "#D97706", "#7C3AED", "#DC2626"]


def measurement_to_east_north(measurement, sensor_configs):
    sensor_id = measurement["sensor_id"]
    if sensor_id in ("radar", "camera") and measurement["range_m"] is not None:
        sensor_position = sensor_configs[sensor_id].get("pos_ned", [0.0, 0.0])
        north, east = polar_to_cartesian(
            measurement["range_m"],
            measurement["bearing_rad"],
            sensor_position,
        )
        return east, north

    if sensor_id == "ais" and measurement["north_m"] is not None:
        return measurement["east_m"], measurement["north_m"]

    return None


def truth_track(scenario_data, target_id=0):
    truth = np.array(scenario_data["ground_truth"][str(target_id)], dtype=float)
    return truth[:, 0], truth[:, 1], truth[:, 2]


def plot_tracking_result(
    scenario_data,
    run_histories,
    output_name,
    title,
    ais_dropout=None,
    target_id=0,
):
    """
    Save a two-panel tracking plot.

    Left: NED scene, truth, measurements, and EKF estimates.
    Right: position error over time for each EKF run.
    """
    TRACKING_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TRACKING_OUTPUT_DIR / output_name

    sensor_configs = scenario_data["sensor_configs"]
    camera_pos = np.array(sensor_configs["camera"]["pos_ned"], dtype=float)
    truth_t, truth_n, truth_e = truth_track(scenario_data, target_id)

    fig, (ax_scene, ax_error) = plt.subplots(1, 2, figsize=(14, 6))

    ax_scene.plot(truth_e, truth_n, color="#111827", lw=2.2, label="Ground truth")
    ax_scene.plot(truth_e[0], truth_n[0], "o", color="#111827", ms=7)
    ax_scene.plot(truth_e[-1], truth_n[-1], "D", color="#111827", ms=6)

    radar_range = sensor_configs["radar"]["range_m"]
    ax_scene.add_patch(
        plt.Circle(
            (0.0, 0.0),
            radar_range,
            fill=False,
            color=SENSOR_COLORS["radar"],
            ls="--",
            lw=1.0,
            alpha=0.35,
        )
    )
    ax_scene.plot(0.0, 0.0, "^", color=SENSOR_COLORS["radar"], ms=9, label="Radar")

    camera_range = sensor_configs["camera"]["range_m"]
    camera_boresight = sensor_configs["camera"]["boresight_deg"]
    ax_scene.add_patch(
        Arc(
            (camera_pos[1], camera_pos[0]),
            2 * camera_range,
            2 * camera_range,
            theta1=90 - camera_boresight - 90,
            theta2=90 - camera_boresight + 90,
            color=SENSOR_COLORS["camera"],
            lw=1.0,
            ls="--",
            alpha=0.4,
        )
    )
    ax_scene.plot(
        camera_pos[1],
        camera_pos[0],
        "s",
        color=SENSOR_COLORS["camera"],
        ms=8,
        label="Camera",
    )

    for sensor_id in ("radar", "camera", "ais"):
        xs = []
        ys = []
        for measurement in scenario_data["measurements"]:
            if measurement["sensor_id"] != sensor_id or measurement["is_false_alarm"]:
                continue
            point = measurement_to_east_north(measurement, sensor_configs)
            if point is None:
                continue
            xs.append(point[0])
            ys.append(point[1])
        if xs:
            ax_scene.plot(
                xs,
                ys,
                ".",
                color=SENSOR_COLORS[sensor_id],
                alpha=0.22,
                ms=4,
                label=f"{sensor_id.upper()} detections",
            )

    for index, (label, history) in enumerate(run_histories.items()):
        estimates = history["estimates"]
        times = history["times"]
        truth = history["truth"]
        if len(estimates) == 0:
            continue

        color = RUN_COLORS[index % len(RUN_COLORS)]
        ax_scene.plot(
            estimates[:, 1],
            estimates[:, 0],
            "-",
            color=color,
            lw=1.6,
            alpha=0.9,
            label=f"{label} estimate",
        )

        errors = np.linalg.norm(estimates - truth, axis=1)
        ax_error.plot(times, errors, color=color, lw=1.6, label=label)

    if ais_dropout is not None:
        ax_error.axvspan(*ais_dropout, color="red", alpha=0.12, label="AIS dropout")

    ax_scene.set_xlabel("East [m]")
    ax_scene.set_ylabel("North [m]")
    ax_scene.set_title("NED Tracking Scene")
    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.grid(True, alpha=0.25)
    ax_scene.legend(fontsize=7, loc="best")

    ax_error.set_xlabel("Time [s]")
    ax_error.set_ylabel("Position error [m]")
    ax_error.set_title("Tracking Error")
    ax_error.grid(True, alpha=0.25)
    ax_error.legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
