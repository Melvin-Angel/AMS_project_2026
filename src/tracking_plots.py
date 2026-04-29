import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

from src.tracking_common import TRACKING_OUTPUT_DIR, polar_to_cartesian


SENSOR_COLORS = {
    "radar": "#2E75B6",
    "camera": "#0B6E4F",
    "ais": "#BA7517",
    "gnss": "#7F77DD",
}
RUN_COLORS = ["#111827", "#D97706", "#7C3AED", "#DC2626"]


def dominant_target_id(track):
    target_ids = [target_id for target_id in getattr(track, "assigned_target_ids", []) if target_id >= 0]
    if not target_ids:
        return None
    values, counts = np.unique(target_ids, return_counts=True)
    return int(values[np.argmax(counts)])


def compact_track_labels(track_ids):
    ordered_track_ids = sorted(track_ids)
    return {track_id: index for index, track_id in enumerate(ordered_track_ids)}


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

    radar_range = sensor_configs["radar"]["range_m"]
    ax_scene.add_patch(
        Circle(
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
            zorder=2,
            label=f"{label} estimate",
        )

        errors = np.linalg.norm(estimates - truth, axis=1)
        ax_error.plot(times, errors, color=color, lw=1.6, label=label)

    ax_scene.plot(truth_e, truth_n, color="#111827", lw=2.4, zorder=3, label="Ground truth")
    ax_scene.plot(truth_e[0], truth_n[0], "o", color="#111827", ms=7, zorder=3)
    ax_scene.plot(truth_e[-1], truth_n[-1], "D", color="#111827", ms=6, zorder=3)

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


def plot_t6_tracks(
    scenario_data,
    tracks,
    target_track_ids,
    output_name,
    title,
):
    """Save a Scenario D multi-target association plot."""
    TRACKING_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TRACKING_OUTPUT_DIR / output_name

    sensor_configs = scenario_data["sensor_configs"]
    camera_pos = np.array(sensor_configs["camera"]["pos_ned"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 7))

    radar_range = sensor_configs["radar"]["range_m"]
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            radar_range,
            fill=False,
            color=SENSOR_COLORS["radar"],
            ls="--",
            lw=1.0,
            alpha=0.25,
        )
    )
    ax.plot(0.0, 0.0, "^", color=SENSOR_COLORS["radar"], ms=9, label="Radar")

    camera_range = sensor_configs["camera"]["range_m"]
    camera_boresight = sensor_configs["camera"]["boresight_deg"]
    ax.add_patch(
        Arc(
            (camera_pos[1], camera_pos[0]),
            2 * camera_range,
            2 * camera_range,
            theta1=90 - camera_boresight - 90,
            theta2=90 - camera_boresight + 90,
            color=SENSOR_COLORS["camera"],
            lw=1.0,
            ls="--",
            alpha=0.35,
        )
    )
    ax.plot(camera_pos[1], camera_pos[0], "s",
            color=SENSOR_COLORS["camera"], ms=8, label="Camera")

    best_track_ids = set(target_track_ids.values())
    display_labels = compact_track_labels(best_track_ids)
    for track in tracks:
        if track.track_id not in best_track_ids:
            continue
        if len(track.history) < 2:
            continue
        history = np.array([record[1] for record in track.history])
        target_id = None
        for candidate_target, candidate_track in target_track_ids.items():
            if candidate_track == track.track_id:
                target_id = candidate_target
                break
        if target_id is None:
            continue
        color = RUN_COLORS[int(target_id) % len(RUN_COLORS)]
        ax.plot(history[:, 1], history[:, 0], "-", color=color, lw=2.2,
                label=f"track {display_labels[track.track_id]} -> target {target_id}")

    for target_index, target_id in enumerate(sorted(scenario_data["ground_truth"], key=int)):
        _, north, east = truth_track(scenario_data, target_id=int(target_id))
        color = RUN_COLORS[target_index % len(RUN_COLORS)]
        ax.plot(east, north, "--", color=color, lw=1.9, alpha=0.8,
                zorder=3, label=f"truth {target_id}")
        ax.plot(east[0], north[0], "o", color=color, ms=5, zorder=3)
        ax.plot(east[-1], north[-1], "D", color=color, ms=5, zorder=3)

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_t7_metrics(
    scenario_data,
    metrics,
    tracks,
    output_name,
    title,
):
    """Save T7 MOTP/cardinality metrics and final confirmed track paths."""
    TRACKING_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TRACKING_OUTPUT_DIR / output_name

    times = np.array([item["time"] for item in metrics])
    motp = np.array([
        np.nan if item["motp"] is None else item["motp"]
        for item in metrics
    ])
    ce = np.array([item["ce"] for item in metrics])
    confirmed = np.array([item["confirmed"] for item in metrics])
    truth_counts = np.array([item["truth"] for item in metrics])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_scene, ax_motp = axes[0]
    ax_ce, ax_count = axes[1]

    displayed_tracks = [
        track for track in tracks
        if track.status in ("confirmed", "coasting") and len(track.history) >= 2
    ]
    display_labels = compact_track_labels(track.track_id for track in displayed_tracks)

    for track in displayed_tracks:
        history = np.array([record[1] for record in track.history])
        target_id = dominant_target_id(track)
        color = RUN_COLORS[int(target_id) % len(RUN_COLORS)] if target_id is not None else RUN_COLORS[track.track_id % len(RUN_COLORS)]
        ax_scene.plot(history[:, 1], history[:, 0], "-", color=color, lw=1.8,
                      zorder=2,
                      label=f"track {display_labels[track.track_id]} -> truth {target_id}" if target_id is not None else f"track {display_labels[track.track_id]}")

    for target_index, target_id in enumerate(sorted(scenario_data["ground_truth"], key=int)):
        _, north, east = truth_track(scenario_data, target_id=int(target_id))
        color = RUN_COLORS[target_index % len(RUN_COLORS)]
        ax_scene.plot(east, north, "--", color=color, lw=1.6, alpha=0.75,
                      zorder=3, label=f"truth {target_id}")
        ax_scene.plot(east[0], north[0], "o", color=color, ms=5, zorder=3)
        ax_scene.plot(east[-1], north[-1], "D", color=color, ms=5, zorder=3)

    ax_scene.set_xlabel("East [m]")
    ax_scene.set_ylabel("North [m]")
    ax_scene.set_title("Final Confirmed/Coasting Tracks")
    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.grid(True, alpha=0.25)
    ax_scene.legend(fontsize=7, loc="best", ncol=2)

    ax_motp.plot(times, motp, color="#111827", lw=1.6)
    ax_motp.set_xlabel("Time [s]")
    ax_motp.set_ylabel("MOTP [m]")
    ax_motp.set_title("Multiple Object Tracking Precision")
    ax_motp.grid(True, alpha=0.25)

    ax_ce.step(times, ce, where="post", color="#D97706", lw=1.6)
    ax_ce.set_xlabel("Time [s]")
    ax_ce.set_ylabel("Cardinality Error")
    ax_ce.set_title("Cardinality Error")
    ax_ce.grid(True, alpha=0.25)

    ax_count.step(times, truth_counts, where="post", color="#111827", lw=1.6,
                  label="true targets")
    ax_count.step(times, confirmed, where="post", color="#7C3AED", lw=1.6,
                  label="confirmed tracks")
    ax_count.set_xlabel("Time [s]")
    ax_count.set_ylabel("Count")
    ax_count.set_title("Confirmed Tracks vs Active Truth")
    ax_count.grid(True, alpha=0.25)
    ax_count.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
