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
    SENSOR_ORDER,
    group_measurements_by_time,
    load_scenario,
    noise_covariance_from_config,
    polar_position_covariance,
    polar_to_cartesian,
    truth_position_at,
)
from src.tracking_plots import plot_t6_tracks
# from src.t7_track_management import compute_MOTP, compute_CE


ASSIGNMENT_PENALTY = 1.0e6


@dataclass
class Track:
    track_id: int
    tracker: EKFTracker
    last_time: float
    missed_detections: int = 0
    total_assignments: int = 1
    assigned_target_ids: list = field(default_factory=list)
    history: list = field(default_factory=list)
    confirmed: bool = False
    confirmation_count: int = 0

class DataAssociator:
    def __init__(self, gate_probability=0.99):
        self.gate_probability = gate_probability
        self.gate_threshold = CHI2_99_THRESHOLDS[2]  # Default for 2D measurements

    def mahalanobis_gate(self, track, measurement, coord_manager, R_by_sensor):
        sensor_id = measurement["sensor_id"]
        z = detection_vector(measurement)
        h = coord_manager.compute_h(track.tracker.x, sensor_id)
        H = coord_manager.compute_jacobian(track.tracker.x, sensor_id)
        R = R_by_sensor[sensor_id]
        innovation = z - h
        innovation[1] = (innovation[1] + np.pi) % (2.0 * np.pi) - np.pi
        S = H @ track.tracker.P @ H.T + R
        distance_squared = float(innovation.T @ np.linalg.solve(S, innovation))
        return distance_squared, h, H, R
    
    
    def associate_gnn(self, tracks, detections, coord_manager, R_by_sensor, gate_probability=0.99):
        """
        Global Nearest Neighbor over track/sensor slots and all detections.

        A row represents one possible sensor contribution to one track. This lets
        the same track receive one radar and one camera update at the same scan,
        while each detection can still be assigned at most once globally.
        """
        if not tracks or not detections:
            return [], set(range(len(detections)))

        available_sensors = sorted(
            {detection["sensor_id"] for detection in detections},
            key=SENSOR_ORDER.index,
        )
        rows = [
            (track_index, sensor_id)
            for track_index, _ in enumerate(tracks)
            for sensor_id in available_sensors
        ]

        cost = np.full((len(rows), len(detections)), ASSIGNMENT_PENALTY)
        gate_threshold = CHI2_99_THRESHOLDS[2]
        gated_distances = {}

        for row_index, (track_index, row_sensor) in enumerate(rows):
            track = tracks[track_index]
            for detection_index, detection in enumerate(detections):
                if detection["sensor_id"] != row_sensor:
                    continue
                distance_squared, h, H, R = self.mahalanobis_gate(
                    track,
                    detection,
                    coord_manager,
                    R_by_sensor,
                )
                if distance_squared <= gate_threshold:
                    cost[row_index, detection_index] = distance_squared
                    gated_distances[(row_index, detection_index)] = (distance_squared, h, H, R)

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

            distance_squared, h, H, R = gated_distances[(row_index, detection_index)]
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

        unmatched_detections = set(range(len(detections))) - assigned_detections
        return assignments, unmatched_detections


# ****************************************************************************************************************
def detection_vector(measurement):
        """Convert measurement to vector form"""
        return np.array([measurement["range_m"], measurement["bearing_rad"]])

def predict_tracks_to_time(tracks, time_s):
    for track in tracks:
        dt = time_s - track.last_time
        if dt > 0:
            track.tracker.predict(dt)
            track.last_time = time_s


def apply_assignments(tracks, detections, assignments, time_s):
    assigned_track_indices = set()
    false_assignments = 0
    true_assignments = 0

    for assignment in sorted(
        assignments,
        key=lambda item: SENSOR_ORDER.index(detections[item["detection_index"]]["sensor_id"]),
    ):
        track = tracks[assignment["track_index"]]
        detection = detections[assignment["detection_index"]]
        z = detection_vector(detection)
        track.tracker.update(z, assignment["h"], assignment["H"], assignment["R"])
        track.missed_detections = 0
        track.total_assignments += 1
        track.assigned_target_ids.append(detection["target_id"])
        track.history.append((time_s, track.tracker.x[:2].copy()))
        assigned_track_indices.add(assignment["track_index"])

        if detection["is_false_alarm"]:
            false_assignments += 1
        else:
            true_assignments += 1

    return assigned_track_indices, true_assignments, false_assignments


def dominant_target_id(track):
    """
    Determine the dominant target ID for a track based on most frequent assignments.
    """
    target_ids = [target_id for target_id in track.assigned_target_ids if target_id >= 0]
    if not target_ids:
        return None
    values, counts = np.unique(target_ids, return_counts=True)
    return int(values[np.argmax(counts)])

# ****************************************************************************************************************

def validate_tracks(tracks, scenario_data, min_assignments=6):
    mature_tracks = [track for track in tracks if track.total_assignments >= min_assignments]
    target_errors = {}
    target_track_ids = {}

    for track in mature_tracks:
        target_id = dominant_target_id(track)
        if target_id is None:
            continue

        errors = []
        for time_s, estimate in track.history:
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

def initialise_track_from_detection(
    track_id,
    measurement,
    coord_manager,
    R_by_sensor,
):
    sensor_id = measurement["sensor_id"]
    sensor_position = coord_manager.get_sensor_position(sensor_id)
    z = detection_vector(measurement)
    position = polar_to_cartesian(z[0], z[1], sensor_position)
    position_covariance = polar_position_covariance(z[0], z[1], R_by_sensor[sensor_id])

    x_initial = np.array([position[0], position[1], 0.0, 0.0])
    P_initial = np.zeros((4, 4))
    P_initial[:2, :2] = position_covariance
    P_initial[2:, 2:] = np.eye(2) * 100.0

    tracker = EKFTracker(x_initial=x_initial, P_initial=P_initial, sigma_a=0.05)
    track = Track(track_id=track_id, tracker=tracker, last_time=measurement["time"])
    track.assigned_target_ids.append(measurement["target_id"])
    track.history.append((measurement["time"], tracker.x[:2].copy()))
    return track


# ****************************************************************************************************************

def run_t6_association(scenario_path=SCENARIO_D_PATH, make_plot=True):
    print("Starting T6 gating and data association (Scenario D)...")

    scenario_data = load_scenario(scenario_path)
    coord_manager = CoordinateFrameManager(
        camera_pos=scenario_data["sensor_configs"]["camera"]["pos_ned"],
    )
    R_by_sensor = {
        sensor_id: noise_covariance_from_config(scenario_data["sensor_configs"][sensor_id])
        for sensor_id in SENSOR_ORDER
    }

    measurements = [
        measurement for measurement in scenario_data["measurements"]
        if measurement["sensor_id"] in SENSOR_ORDER
    ]
    scans = group_measurements_by_time(measurements)

    tracks = []
    next_track_id = 0
    max_missed = 6
    assignment_count = 0
    true_assignment_count = 0
    false_assignment_count = 0
    initiated_count = 0
    unmatched_detection_count = 0
    missed_track_count = 0
    sensor_skip_count = {sensor_id: 0 for sensor_id in SENSOR_ORDER}

    # Initialize DataAssociator
    associator = DataAssociator(gate_probability=0.99)

    for time_s in sorted(scans):
        scan_detections = scans[time_s]
        sensor_available = {
            sensor_id: any(detection["sensor_id"] == sensor_id for detection in scan_detections)
            for sensor_id in SENSOR_ORDER
        }
        for sensor_id, available in sensor_available.items():
            if not available:
                sensor_skip_count[sensor_id] += 1

        predict_tracks_to_time(tracks, time_s)

        assignments, unmatched_detections = associator.associate_gnn(
            tracks,
            scan_detections,
            coord_manager,
            R_by_sensor,
        )
        assignment_count += len(assignments)
        true_assigned, false_assigned = 0, 0
        assigned_track_indices, true_assigned, false_assigned = apply_assignments(
            tracks,
            scan_detections,
            assignments,
            time_s,
        )
        true_assignment_count += true_assigned
        false_assignment_count += false_assigned

        for track_index, track in enumerate(tracks):
            if track_index not in assigned_track_indices:
                track.missed_detections += 1
                missed_track_count += 1

        for detection_index in sorted(unmatched_detections):
            detection = scan_detections[detection_index]
            track = initialise_track_from_detection(
                next_track_id,
                detection,
                coord_manager,
                R_by_sensor,
            )
            tracks.append(track)
            next_track_id += 1
            initiated_count += 1
            unmatched_detection_count += 1

        tracks = [
            track for track in tracks
            if track.missed_detections <= max_missed or track.total_assignments >= 4
        ]

    mature_tracks, target_errors, target_track_ids = validate_tracks(tracks, scenario_data)
    covered_targets = sorted(target_errors)
    mean_target_rmse = (
        float(np.mean(list(target_errors.values()))) if target_errors else None
    )

    plot_path = None
    if make_plot:
        plot_path = plot_t6_tracks(
            scenario_data,
            mature_tracks,
            target_track_ids,
            "t6_gating_association.png",
            "T6 Gating + GNN Association - Scenario D",
        )

    print("\nT6 validation summary")
    print(f"Scans processed                    : {len(scans)}")
    print(f"Gate threshold chi2_2(P_G=0.99)    : {CHI2_99_THRESHOLDS[2]:.3f}")
    print(f"Assignments accepted               : {assignment_count}")
    print(f"True detection assignments         : {true_assignment_count}")
    print(f"False-alarm assignments            : {false_assignment_count}")
    print(f"Unmatched detections -> initiation : {unmatched_detection_count}")
    print(f"Tracks initiated                   : {initiated_count}")
    print(f"Tracks remaining                   : {len(tracks)}")
    print(f"Mature tracks (>=6 assignments)    : {len(mature_tracks)}")
    print(f"Missed-detection flags             : {missed_track_count}")
    print(f"Sensor skipped scans               : {sensor_skip_count}")
    print(f"Covered true targets               : {covered_targets}")
    if mean_target_rmse is not None:
        print(f"Mean best-track target RMSE        : {mean_target_rmse:.2f} m")
        for target_id in covered_targets:
            print(
                f"  target {target_id}: track {target_track_ids[target_id]} "
                f"RMSE={target_errors[target_id]:.2f} m"
            )
    
    # print(f"MOTP avg            : {MOTP_avg:.2f}")
    print(f"CE avg              : {CE_avg:.2f}")

    if plot_path is not None:
        print(f"Tracking plot                      : {plot_path}")

    return {
        "assignments": assignment_count,
        "true_assignments": true_assignment_count,
        "false_assignments": false_assignment_count,
        "initiated_tracks": initiated_count,
        "remaining_tracks": len(tracks),
        "mature_tracks": len(mature_tracks),
        "sensor_skip_count": sensor_skip_count,
        "covered_targets": covered_targets,
        "target_errors": target_errors,
        "mean_target_rmse": mean_target_rmse,
        "plot_path": plot_path,
    }

# ****************************************************************************************************************

if __name__ == "__main__":
    run_t6_association()



