import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coordinate_manager import CoordinateFrameManager
from src.ekf_tracker import EKFTracker
from src.tracking_common import (
    CHI2_99_THRESHOLDS,
    SCENARIO_D_PATH,
    SCENARIO_E_PATH,
    group_measurements_by_time,
    load_scenario,
    noise_covariance_from_config,
    polar_position_covariance,
    polar_to_cartesian,
    truth_position_at,
)
from src.tracking_plots import dominant_target_id, plot_t7_metrics
from src.tracking_common import TRACKING_OUTPUT_DIR


T7_SENSOR_ORDER = ("radar", "camera", "ais")
ASSIGNMENT_PENALTY = 1.0e6


@dataclass
class ManagedTrack:
    track_id: int
    tracker: EKFTracker
    last_time: float
    created_time: float
    status: str = "tentative"
    hits: list = field(default_factory=lambda: [True])
    missed_detections: int = 0
    total_assignments: int = 1
    assigned_target_ids: list = field(default_factory=list)
    history: list = field(default_factory=list)
    last_detection_position: np.ndarray | None = None
    last_detection_time: float | None = None


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def detection_vector(coord_manager, measurement):
    if measurement["sensor_id"] == "ais":
        return coord_manager.compute_ais_measurement_from_report(measurement)
    return np.array([measurement["range_m"], measurement["bearing_rad"]])


def detection_position(coord_manager, measurement):
    sensor_id = measurement["sensor_id"]
    if sensor_id == "ais":
        return np.array([measurement["north_m"], measurement["east_m"]])

    sensor_position = coord_manager.get_sensor_position(sensor_id)
    return polar_to_cartesian(
        measurement["range_m"],
        measurement["bearing_rad"],
        sensor_position,
    )


def measurement_covariance(coord_manager, R_by_sensor, track, measurement):
    sensor_id = measurement["sensor_id"]
    if sensor_id == "ais":
        return coord_manager.get_noise_covariance("ais", track.tracker.x, measurement["time"])
    return R_by_sensor[sensor_id]


def initial_covariance_from_detection(coord_manager, R_by_sensor, measurement):
    sensor_id = measurement["sensor_id"]
    if sensor_id == "ais":
        sigma = coord_manager.sigma_ais
        return np.eye(2) * sigma**2

    z = np.array([measurement["range_m"], measurement["bearing_rad"]])
    return polar_position_covariance(z[0], z[1], R_by_sensor[sensor_id])


def initialise_track(track_id, measurement, coord_manager, R_by_sensor):
    position = detection_position(coord_manager, measurement)
    position_covariance = initial_covariance_from_detection(
        coord_manager,
        R_by_sensor,
        measurement,
    )

    x_initial = np.array([position[0], position[1], 0.0, 0.0])
    P_initial = np.zeros((4, 4))
    P_initial[:2, :2] = position_covariance
    P_initial[2:, 2:] = np.eye(2) * 16.0

    tracker = EKFTracker(x_initial=x_initial, P_initial=P_initial, sigma_a=0.05)
    track = ManagedTrack(
        track_id=track_id,
        tracker=tracker,
        last_time=measurement["time"],
        created_time=measurement["time"],
        assigned_target_ids=[measurement["target_id"]],
        history=[(measurement["time"], tracker.x[:2].copy(), "tentative")],
        last_detection_position=position,
        last_detection_time=measurement["time"],
    )
    return track


def build_event_scans(scenario_data):
    events = [
        measurement for measurement in scenario_data["measurements"]
        if measurement["sensor_id"] in (*T7_SENSOR_ORDER, "gnss")
    ]
    grouped = group_measurements_by_time(events)
    return grouped


def configure_tracking(scenario_data):
    coord_manager = CoordinateFrameManager(
        camera_pos=scenario_data["sensor_configs"]["camera"]["pos_ned"],
        sigma_ais=scenario_data["sensor_configs"]["ais"]["sigma_pos_m"],
    )
    R_by_sensor = {
        sensor_id: noise_covariance_from_config(scenario_data["sensor_configs"][sensor_id])
        for sensor_id in ("radar", "camera")
    }
    return coord_manager, R_by_sensor


def predict_tracks_to_time(tracks, time_s):
    for track in tracks:
        dt = time_s - track.last_time
        if dt > 0:
            track.tracker.predict(dt)
            track.last_time = time_s
            if track.status == "coasting":
                track.history.append((time_s, track.tracker.x[:2].copy(), "coasting"))


def mahalanobis_gate(track, measurement, coord_manager, R_by_sensor):
    sensor_id = measurement["sensor_id"]
    if track.status == "tentative" and track.last_detection_time is not None:
        measured_position = detection_position(coord_manager, measurement)
        dt = measurement["time"] - track.last_detection_time
        if dt > 0:
            implied_speed = np.linalg.norm(
                (measured_position - track.last_detection_position) / dt
            )
            if implied_speed > 15.0:
                return np.inf, None, None, None

    z = detection_vector(coord_manager, measurement)
    h = coord_manager.compute_h(track.tracker.x, sensor_id, measurement["time"])
    H = coord_manager.compute_jacobian(track.tracker.x, sensor_id, measurement["time"])
    R = measurement_covariance(coord_manager, R_by_sensor, track, measurement)
    innovation = z - h
    innovation[1] = wrap_angle(innovation[1])
    S = H @ track.tracker.P @ H.T + R
    distance_squared = float(innovation.T @ np.linalg.solve(S, innovation))
    return distance_squared, h, H, R


def associate_gnn(tracks, detections, coord_manager, R_by_sensor):
    if not tracks or not detections:
        return [], set(range(len(detections)))

    available_sensors = sorted(
        {detection["sensor_id"] for detection in detections},
        key=T7_SENSOR_ORDER.index,
    )
    rows = [
        (track_index, sensor_id)
        for track_index, _ in enumerate(tracks)
        for sensor_id in available_sensors
    ]

    cost = np.full((len(rows), len(detections)), ASSIGNMENT_PENALTY)
    gated = {}
    gate_threshold = CHI2_99_THRESHOLDS[2]

    for row_index, (track_index, sensor_id) in enumerate(rows):
        track = tracks[track_index]
        for detection_index, detection in enumerate(detections):
            if detection["sensor_id"] != sensor_id:
                continue
            distance_squared, h, H, R = mahalanobis_gate(
                track,
                detection,
                coord_manager,
                R_by_sensor,
            )
            if distance_squared <= gate_threshold:
                cost[row_index, detection_index] = distance_squared
                gated[(row_index, detection_index)] = (distance_squared, h, H, R)

    row_indices, detection_indices = linear_sum_assignment(cost)
    assignments = []
    assigned_detections = set()
    used_track_sensor_slots = set()

    for row_index, detection_index in zip(row_indices, detection_indices):
        if cost[row_index, detection_index] >= ASSIGNMENT_PENALTY:
            continue

        track_index, sensor_id = rows[row_index]
        slot = (track_index, sensor_id)
        if slot in used_track_sensor_slots:
            continue

        distance_squared, h, H, R = gated[(row_index, detection_index)]
        assignments.append({
            "track_index": track_index,
            "detection_index": detection_index,
            "distance_squared": distance_squared,
            "h": h,
            "H": H,
            "R": R,
        })
        assigned_detections.add(detection_index)
        used_track_sensor_slots.add(slot)

    return assignments, set(range(len(detections))) - assigned_detections


def update_track_lifecycle(track, hit, M, N, time_s):
    track.hits.append(hit)
    track.hits = track.hits[-N:]

    if hit:
        track.missed_detections = 0
        old_enough = (time_s - track.created_time) >= 3.0
        if track.status in ("tentative", "coasting") and sum(track.hits) >= M and old_enough:
            track.status = "confirmed"
    else:
        track.missed_detections += 1
        if track.status == "confirmed":
            track.status = "coasting"


def estimate_initial_velocity(track, detection_position_value, detection_time):
    if track.last_detection_position is None or track.last_detection_time is None:
        return

    dt = detection_time - track.last_detection_time
    if dt <= 0:
        return

    velocity = (detection_position_value - track.last_detection_position) / dt
    track.tracker.x[2:] = velocity
    track.tracker.P[2:, 2:] = np.eye(2) * 25.0


def apply_assignments(
    tracks,
    detections,
    assignments,
    coord_manager,
    M,
    N,
):
    assigned_track_indices = set()
    false_assignments = 0

    for assignment in sorted(
        assignments,
        key=lambda item: T7_SENSOR_ORDER.index(detections[item["detection_index"]]["sensor_id"]),
    ):
        track = tracks[assignment["track_index"]]
        detection = detections[assignment["detection_index"]]
        time_s = detection["time"]

        if track.total_assignments == 1:
            estimate_initial_velocity(track, detection_position(coord_manager, detection), time_s)

        z = detection_vector(coord_manager, detection)
        track.tracker.update(z, assignment["h"], assignment["H"], assignment["R"])
        track.total_assignments += 1
        track.assigned_target_ids.append(detection["target_id"])
        track.last_detection_position = detection_position(coord_manager, detection)
        track.last_detection_time = time_s
        update_track_lifecycle(track, hit=True, M=M, N=N, time_s=time_s)
        track.history.append((time_s, track.tracker.x[:2].copy(), track.status))
        assigned_track_indices.add(assignment["track_index"])

        if detection["is_false_alarm"]:
            false_assignments += 1

    return assigned_track_indices, false_assignments


def delete_dead_tracks(tracks, K_del):
    alive = []
    deleted_count = 0
    for track in tracks:
        if track.missed_detections > K_del:
            deleted_count += 1
        else:
            alive.append(track)
    return alive, deleted_count


def merge_duplicate_tracks(tracks, merge_threshold=9.210):
    if len(tracks) < 2:
        return tracks, 0

    removed = set()
    merges = 0
    for i, track_i in enumerate(tracks):
        if i in removed or track_i.status not in ("confirmed", "coasting"):
            continue
        for j in range(i + 1, len(tracks)):
            track_j = tracks[j]
            if j in removed or track_j.status not in ("confirmed", "coasting"):
                continue
            delta = track_i.tracker.x[:2] - track_j.tracker.x[:2]
            covariance = track_i.tracker.P[:2, :2] + track_j.tracker.P[:2, :2]
            distance = float(delta.T @ np.linalg.solve(covariance, delta))
            if distance <= merge_threshold:
                keep, drop = (i, j)
                if tracks[j].total_assignments > tracks[i].total_assignments:
                    keep, drop = (j, i)
                removed.add(drop)
                tracks[keep].assigned_target_ids.extend(tracks[drop].assigned_target_ids)
                tracks[keep].history.extend(tracks[drop].history)
                merges += 1

    return [track for index, track in enumerate(tracks) if index not in removed], merges


def validate_tracks(tracks, scenario_data, min_assignments=6):
    mature_tracks = [track for track in tracks if track.total_assignments >= min_assignments]
    target_errors = {}
    target_track_ids = {}

    for track in mature_tracks:
        target_id = dominant_target_id(track)
        if target_id is None:
            continue

        errors = []
        for record in track.history:
            time_s, estimate = record[:2]
            truth = truth_position_at(scenario_data["ground_truth"], target_id, time_s)
            errors.append(estimate - truth)

        if not errors:
            continue

        rmse = float(np.sqrt(np.mean(np.sum(np.array(errors) ** 2, axis=1))))
        current_best = target_errors.get(target_id)
        if current_best is None or rmse < current_best:
            target_errors[target_id] = rmse
            target_track_ids[target_id] = track.track_id

    return mature_tracks, target_errors, target_track_ids


def active_truth_positions(scenario_data, time_s):
    active = []
    for target_id, rows in scenario_data["ground_truth"].items():
        truth = np.array(rows, dtype=float)
        if truth[0, 0] <= time_s <= truth[-1, 0]:
            active.append((int(target_id), truth_position_at(scenario_data["ground_truth"], target_id, time_s)))
    return active


def match_confirmed_tracks_to_truth(tracks, scenario_data, time_s):
    confirmed = [
        track for track in tracks
        if track.status in ("confirmed", "coasting")
    ]
    active_truths = active_truth_positions(scenario_data, time_s)
    ce = abs(len(confirmed) - len(active_truths))

    if not confirmed or not active_truths:
        return confirmed, active_truths, ce, None, []

    cost = np.zeros((len(confirmed), len(active_truths)))
    for i, track in enumerate(confirmed):
        for j, (_, truth_pos) in enumerate(active_truths):
            cost[i, j] = np.linalg.norm(track.tracker.x[:2] - truth_pos)

    row_indices, col_indices = linear_sum_assignment(cost)
    matched_errors = cost[row_indices, col_indices]
    matches = [
        {
            "track_id": int(confirmed[i].track_id),
            "target_id": int(active_truths[j][0]),
            "error_m": float(cost[i, j]),
        }
        for i, j in zip(row_indices, col_indices)
    ]
    return confirmed, active_truths, ce, float(np.mean(matched_errors)), matches


def compute_motp_ce(tracks, scenario_data, time_s):
    confirmed, active_truths, ce, motp, _ = match_confirmed_tracks_to_truth(
        tracks,
        scenario_data,
        time_s,
    )
    return motp, ce, len(confirmed), len(active_truths)


def write_t7_debug_report(
    scenario_name,
    metrics,
    tracks,
    target_errors,
    target_track_ids,
    plot_path,
    scan_debug,
    M,
    N,
    K_del,
):
    TRACKING_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TRACKING_OUTPUT_DIR / f"t7_track_management_{scenario_name}_debug.json"

    final_tracks = []
    for track in sorted(tracks, key=lambda item: item.track_id):
        final_tracks.append({
            "track_id": int(track.track_id),
            "status": track.status,
            "total_assignments": int(track.total_assignments),
            "missed_detections": int(track.missed_detections),
            "history_length": int(len(track.history)),
            "dominant_target_id": dominant_target_id(track),
        })

    payload = {
        "scenario": scenario_name,
        "parameters": {
            "M": M,
            "N": N,
            "K_del": K_del,
        },
        "plot_path": str(plot_path) if plot_path is not None else None,
        "metrics": metrics,
        "scan_debug": scan_debug,
        "target_errors": {str(target_id): float(error) for target_id, error in target_errors.items()},
        "target_track_ids": {str(target_id): int(track_id) for target_id, track_id in target_track_ids.items()},
        "final_tracks": final_tracks,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return output_path


def run_t7_scenario(
    scenario_path,
    M=2,
    N=5,
    K_del=7,
    make_plot=True,
):
    scenario_data = load_scenario(scenario_path)
    scenario_name = scenario_data["scenario_name"]
    print(f"Starting T7 track management (Scenario {scenario_name})...")

    coord_manager, R_by_sensor = configure_tracking(scenario_data)
    scans = build_event_scans(scenario_data)

    tracks = []
    next_track_id = 0
    deleted_count = 0
    duplicate_merges = 0
    false_assignments = 0
    metrics = []
    scan_debug = []

    for time_s in sorted(scans):
        scan_events = scans[time_s]
        for event in scan_events:
            if event["sensor_id"] == "gnss":
                coord_manager.update_vessel_position(event["north_m"], event["east_m"], time_s)

        detections = [
            event for event in scan_events
            if event["sensor_id"] in T7_SENSOR_ORDER
        ]

        predict_tracks_to_time(tracks, time_s)
        assignments, unmatched_detections = associate_gnn(
            tracks,
            detections,
            coord_manager,
            R_by_sensor,
        )
        assigned_track_indices, false_count = apply_assignments(
            tracks,
            detections,
            assignments,
            coord_manager,
            M,
            N,
        )
        false_assignments += false_count

        for track_index, track in enumerate(tracks):
            if track_index not in assigned_track_indices:
                update_track_lifecycle(track, hit=False, M=M, N=N, time_s=time_s)
                track.history.append((time_s, track.tracker.x[:2].copy(), track.status))

        for detection_index in sorted(unmatched_detections):
            detection = detections[detection_index]
            tracks.append(initialise_track(next_track_id, detection, coord_manager, R_by_sensor))
            next_track_id += 1

        tracks, deleted_now = delete_dead_tracks(tracks, K_del)
        deleted_count += deleted_now
        tracks, merged_now = merge_duplicate_tracks(tracks)
        duplicate_merges += merged_now

        confirmed, active_truths, ce, motp, matches = match_confirmed_tracks_to_truth(
            tracks,
            scenario_data,
            time_s,
        )
        metrics.append({
            "time": time_s,
            "motp": motp,
            "ce": ce,
            "confirmed": len(confirmed),
            "truth": len(active_truths),
        })
        scan_debug.append({
            "time": time_s,
            "confirmed_track_ids": [int(track.track_id) for track in confirmed],
            "active_truth_ids": [int(target_id) for target_id, _ in active_truths],
            "matches": matches,
        })

    motp_values = [item["motp"] for item in metrics if item["motp"] is not None]
    ce_values = [item["ce"] for item in metrics]
    avg_motp = float(np.mean(motp_values)) if motp_values else None
    avg_ce = float(np.mean(ce_values)) if ce_values else None

    mature_tracks, target_errors, target_track_ids = validate_tracks(tracks, scenario_data)
    covered_targets = sorted(target_errors)
    mean_target_rmse = (
        float(np.mean(list(target_errors.values()))) if target_errors else None
    )

    plot_path = None
    if make_plot:
        plot_path = plot_t7_metrics(
            scenario_data,
            metrics,
            tracks,
            f"t7_track_management_{scenario_name}.png",
            f"T7 Track Management - Scenario {scenario_name}",
        )

    debug_path = write_t7_debug_report(
        scenario_name,
        metrics,
        tracks,
        target_errors,
        target_track_ids,
        plot_path,
        scan_debug,
        M,
        N,
        K_del,
    )

    print(f"\nT7 Scenario {scenario_name} summary")
    print(f"M-of-N confirmation              : M={M}, N={N}")
    print(f"Deletion threshold K_del         : {K_del}")
    print(f"Tracks initiated                 : {next_track_id}")
    print(f"Tracks deleted                   : {deleted_count}")
    print(f"Duplicate merges                 : {duplicate_merges}")
    print(f"Tracks alive at end              : {len(tracks)}")
    print(f"Confirmed/coasting at end        : {sum(t.status in ('confirmed', 'coasting') for t in tracks)}")
    print(f"False-alarm assignments          : {false_assignments}")
    print(f"Average MOTP                     : {avg_motp:.2f} m" if avg_motp is not None else "Average MOTP                     : n/a")
    print(f"Average Cardinality Error        : {avg_ce:.2f}" if avg_ce is not None else "Average Cardinality Error        : n/a")
    print(f"Tracks mature (>=6 assignments)  : {len(mature_tracks)}")
    print(f"Covered true targets             : {covered_targets}")
    if mean_target_rmse is not None:
        print(f"Mean best-track target RMSE      : {mean_target_rmse:.2f} m")
        for target_id in covered_targets:
            print(
                f"  target {target_id}: track {target_track_ids[target_id]} "
                f"RMSE={target_errors[target_id]:.2f} m"
            )
    if plot_path is not None:
        print(f"Metrics plot                     : {plot_path}")
    print(f"Debug report                     : {debug_path}")

    return {
        "scenario": scenario_name,
        "avg_motp": avg_motp,
        "avg_ce": avg_ce,
        "metrics": metrics,
        "tracks_initiated": next_track_id,
        "tracks_deleted": deleted_count,
        "duplicate_merges": duplicate_merges,
        "tracks_alive": len(tracks),
        "confirmed_alive": sum(t.status in ("confirmed", "coasting") for t in tracks),
        "false_assignments": false_assignments,
        "plot_path": plot_path,
        "debug_path": debug_path,
    }
def run_t7_track_management(make_plot=True):
    scenario_d = run_t7_scenario(SCENARIO_D_PATH, make_plot=make_plot)
    scenario_e = run_t7_scenario(SCENARIO_E_PATH, make_plot=make_plot)
    return {
        "D": scenario_d,
        "E": scenario_e,
    }


if __name__ == "__main__":
    run_t7_track_management()
