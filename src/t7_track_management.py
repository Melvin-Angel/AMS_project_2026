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
from src.tracking_plots import plot_t7_tracks
from src.t6_gating_association import Track, DataAssociator, detection_vector, dominant_target_id, validate_tracks


ASSIGNMENT_PENALTY = 1.0e6


class TrackManager:
    def __init__(self, M=3, N=5, K_del=9, merge_threshold=5.0):
        self.M = M          # Confirmation threshold
        self.N = N          # Confirmation window
        self.K_del = K_del  # max consecutive misses before deletion
        self.tracks: list[Track] = []
        self.next_track_ID = 0
        self.merge_threshold = merge_threshold
    
    def initialise_track(self, measurement, coord_manager, R_by_sensor):
        """"
        Initialize a new track from a detection.
        Tentative: a new EKF instance is spawned for each unassigned detection
        """
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
        # Start with total_assignments=0 so M counts actual EKF updates, not initialization
        track = Track(track_id=self.next_track_ID, tracker=tracker, last_time=measurement["time"], confirmed=False, total_assignments=0)
        track.assigned_target_ids.append(measurement["target_id"])
        track.history.append((measurement["time"], tracker.x[:2].copy()))
        self.next_track_ID += 1
        return track


    def manage_tracks(self, scans, coord_manager, R_by_sensor):
        for time_s in sorted(scans):
            detections = scans[time_s]

            # 1. Predict all tracks to current time
            self.predict_tracks_to_time(time_s)

            # 2. Associate detections to tracks
            assignments, unmatched = self.associate_detections(
                detections, coord_manager, R_by_sensor
            )
            print(f"Time {time_s:.1f}s: {len(assignments)} assignments, {len(unmatched)} unmatched")

            # 3. Apply assignments inline (instead of using T6's apply_assignments)
            assigned_track_indices = set()

            for assignment in assignments:
                track_idx = assignment["track_index"]
                det_idx = assignment["detection_index"]
                track = self.tracks[track_idx]
                detection = detections[det_idx]

                # Apply the EKF update
                z = detection_vector(detection)
                track.tracker.update(z, assignment["h"], assignment["H"], assignment["R"])

                # Update track metadata
                track.missed_detections = 0
                track.total_assignments += 1
                track.assigned_target_ids.append(detection["target_id"])
                track.history.append((time_s, track.tracker.x[:2].copy()))

                assigned_track_indices.add(track_idx)

            # 4. Handle unassigned tracks (missed detections)
            for i, track in enumerate(self.tracks):
                if i not in assigned_track_indices:
                    track.missed_detections += 1
                    print(f"  Track {track.track_id} missed detection ({track.missed_detections}/{self.K_del})")

            # 5. Delete tracks with too many missed detections (keep confirmed tracks)
            self.tracks = [
                track for track in self.tracks
                if track.missed_detections < self.K_del or track.confirmed
            ]

            # 6. Initiate new tracks from unmatched detections
            for idx in unmatched:
                detection = detections[idx]
                track = self.initialise_track(detection, coord_manager, R_by_sensor)
                self.tracks.append(track)
                print(f"  New track {track.track_id} initiated from {detection['sensor_id']} detection")

            # 7. Update confirmation status for all tracks
            for track in self.tracks:
                if not track.confirmed and track.total_assignments >= self.M:
                    track.confirmed = True
                    print(f"  Track {track.track_id} CONFIRMED ({track.total_assignments}/{self.M} assignments)")

            # 8. Merge duplicate tracks (disabled for now to debug)
            # self.merge_duplicates()


    def predict_tracks_to_time(self, time_s):
        for track in self.tracks:
            dt = time_s - track.last_time
            if dt > 0:
                track.tracker.predict(dt)
                track.last_time = time_s

    def associate_detections(self, detections, coord_manager, R_by_sensor):
        associator = DataAssociator(gate_probability=0.99)
        return associator.associate_gnn(
            self.tracks,
            detections,
            coord_manager,
            R_by_sensor,
        )

    def merge_duplicates(self):
        """Merge tracks with overlapping state estimates using Mahalanobis distance."""
        if len(self.tracks) < 2:
            return

        n = len(self.tracks)
        # Build cost matrix with Mahalanobis distances
        cost = np.full((n, n), np.inf)
        
        for i in range(n):
            for j in range(i + 1, n):
                track1 = self.tracks[i]
                track2 = self.tracks[j]
                delta = track1.tracker.x - track2.tracker.x
                cov_sum = track1.tracker.P + track2.tracker.P
                try:
                    distance = np.sqrt(delta.T @ np.linalg.solve(cov_sum, delta))
                    if distance < self.merge_threshold:
                        cost[i, j] = distance
                        cost[j, i] = distance
                except np.linalg.LinAlgError:
                    continue
        
        # Use Hungarian algorithm to find optimal pairing
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
            # Collect unique pairs to merge
            merge_pairs = []
            used = set()
            for i, j in zip(row_ind, col_ind):
                if i != j and cost[i, j] < np.inf and i not in used and j not in used:
                    merge_pairs.append((i, j))
                    used.add(i)
                    used.add(j)
            
            # Merge pairs: keep track with lower covariance trace
            if merge_pairs:
                # Sort by index to avoid modifying list while iterating
                merge_pairs.sort(reverse=True)  # Merge from highest index first
                for i, j in merge_pairs:
                    if i >= len(self.tracks) or j >= len(self.tracks):
                        continue
                    if np.trace(self.tracks[i].tracker.P) < np.trace(self.tracks[j].tracker.P):
                        self.tracks.pop(j)
                    else:
                        self.tracks.pop(i)
        except ValueError:
            return

    def get_confirmed_tracks(self):
        """Return confirmed tracks"""
        return [track for track in self.tracks if track.confirmed]

        

# ****************************************************************************************************************

def compute_MOTP(tracks, scenario_data):
    """"
    Calculate MOTP (Multiple Object Tracking Precision) as mean localization error over all matched track-target pairs.

    Args:
        tracks: List of track objects with history of estimates.
        scenario_data: Dictionary containing ground truth data.

    Returns:
        MOTP_time_series: List of MOTP values over time.
        motp_avg: Scalar average of MOTP.
    """
    total_error = 0.0
    total_matches = 0

    MOTP_time_series = []

    # Group measurements by time to process each scan
    measurements_by_time = group_measurements_by_time(scenario_data["measurements"])

    # Get all unique times from the tracks' history
    all_times = sorted({time_s for track in tracks for time_s, _ in track.history})

    for time_s in all_times:
        # Get estimates and truths at current time
        estimates = []
        truths = []
        for track in tracks:
            for t, estimate in track.history:
                if t == time_s:
                    target_id = dominant_target_id(track)
                    if target_id is not None:
                        truth = truth_position_at(scenario_data["ground_truth"], target_id, time_s)
                        if truth is not None:
                            estimates.append(estimate)
                            truths.append(truth)

        if not estimates or not truths:
            MOTP_time_series.append((time_s, 0.0))
            continue
        # Convert to numpy arrays
        estimates = np.array(estimates)
        truths = np.array(truths)

        # Match tracks to targets using minimum-distance assignment
        cost_matrix = np.linalg.norm(estimates[:, np.newaxis] - truths, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Calculate distances for matched pairs
        matched_distances = cost_matrix[row_ind, col_ind]
        if len(matched_distances) == 0:
            MOTP_time_series.append((time_s, 0.0))
            continue

        # Calculate MOTP for current time
        MOTP_t = np.mean(matched_distances)
        MOTP_time_series.append((time_s, MOTP_t))

        # Accumulate for average
        total_error += np.sum(MOTP_t)
        total_matches += len(matched_distances)
        print(f"time {time_s}: matches={len(matched_distances)}, motp={MOTP_t:.2f}")

    MOTP_avg = total_error / total_matches if total_matches > 0 else 0.0
    return MOTP_time_series, MOTP_avg



# ****************************************************************************************************************
def compute_CE(tracks, scenario_data, time_tol=1e-6):
    """
    Calculate Cardinality Error (CE) as a time series and scalar average.
    CE measures the mean absolute difference between the number of confirmed tracks and 
    the number of active true targets per scan

    Args:
        tracks: List of track objects with history of estimates.
        scenario_data: Dictionary containing ground truth data.

    Returns:
        CE_time_series: List of CE values over time.
        CE_avg: Scalar average of CE.
    """
    total_CE = 0.0
    total_scans = 0
    CE_time_series = []

    def is_time_match(t1, t2):
        return abs(t1 - t2) < time_tol

    # Group measurements by time to process each scan
    measurements_by_time = group_measurements_by_time(scenario_data["measurements"])

    # Preprocess ground truth times for efficiency
    gt_times = {
        target_id: set(row[0] for row in truth_data)
        for target_id, truth_data in scenario_data["ground_truth"].items()
    }

    for time_s in sorted(measurements_by_time.keys()):
        # Count active true targets at this time
        active_targets = sum(
            any(is_time_match(time_s, t) for t in times)
            for times in gt_times.values()
        )

        # Count confirmed tracks at this time
        confirmed_tracks = sum(1 for track in tracks if track.confirmed and is_time_match(track.last_time, time_s))


        # Calculate CE for this scan
        ce = abs(confirmed_tracks - active_targets)
        CE_time_series.append((time_s, ce))
        total_CE += ce
        total_scans += 1

    CE_avg = total_CE / len(CE_time_series) if CE_time_series else None

    return CE_time_series, CE_avg


# ****************************************************************************************************************

def run_t7_track_management(scenario_path=SCENARIO_D_PATH, scenario_name="D", make_plot=True):
    print(f"Starting T7 track management (Scenario {scenario_name})...")

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

    # Use M=2 for confirmation (scenario D: tracks confirmed after 2 EKF updates)
    # Tracks start with total_assignments=0, need 2 assignments to confirm
    # K_del=9 allows tracks to coast through missed detections
    track_manager = TrackManager(M=2, N=5, K_del=9, merge_threshold=100.0)
    track_manager.manage_tracks(scans, coord_manager, R_by_sensor)

    mature_tracks, target_errors, target_track_ids = validate_tracks(
        track_manager.tracks,
        scenario_data
    )
    # Calculate mean RMSE
    covered_targets = sorted(target_errors)
    mean_target_rmse = (
        float(np.mean(list(target_errors.values()))) if target_errors else None
    )

    # evaluate performance
    # MOTP
    MOTP_time_series, MOTP_avg = compute_MOTP(track_manager.tracks, scenario_data)
    # CE
    CE_time_series, CE_avg = compute_CE(track_manager.tracks, scenario_data)

    #  plot results
    plot_path = None
    if make_plot:
        plot_path = plot_t7_tracks(
            scenario_data,
            mature_tracks,
            target_track_ids,
            MOTP_time_series,
            CE_time_series,
            f"t7_track_management_Sc-{scenario_name}.png",
            f"T7 Track Management - Scenario {scenario_name}",
        )

    print("\nT7 validation summary")
    print(f"Scans processed                    : {len(scans)}")
    print(f"Gate threshold chi2_2(P_G=0.99)    : {CHI2_99_THRESHOLDS[2]:.3f}")
    print(f"Tracks remaining                   : {len(track_manager.tracks)}")
    print(f"Mature tracks (>=6 assignments)    : {len(mature_tracks)}")
    print(f"Covered true targets               : {covered_targets}")
    print(f"Confirmed targets                  : {track_manager.get_confirmed_tracks()}")
    print(f"MOTP avg            : {MOTP_avg:.2f}")
    print(f"CE avg              : {CE_avg:.2f}")

    if mean_target_rmse is not None:
        print(f"Mean best-track target RMSE        : {mean_target_rmse:.2f} m")
        for target_id in covered_targets:
            print(
                f"  target {target_id}: track {target_track_ids[target_id]} "
                f"RMSE={target_errors[target_id]:.2f} m"
            )
    
    if plot_path is not None:
        print(f"Tracking plot                      : {plot_path}")

    return 

if __name__ == "__main__":
    # run_t7_track_management(SCENARIO_D_PATH, "D", True)
    run_t7_track_management(SCENARIO_E_PATH, "E", True)