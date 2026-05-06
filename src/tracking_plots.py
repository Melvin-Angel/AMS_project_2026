import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

from src.tracking_common import TRACKING_OUTPUT_DIR, polar_to_cartesian

try:
    import contextily as ctx
    _HAS_CONTEXTILY = True
except ImportError:
    _HAS_CONTEXTILY = False


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


def plot_multi_target_scene(ax, scenario_data, tracks, target_track_ids):
    """
    Plot tracking scene (ground truth + sensors + selected tracks)
    onto an existing matplotlib axis.

    This is shared between T6 and T7 to ensure identical visuals.
    """

    sensor_configs = scenario_data["sensor_configs"]
    camera_pos = np.array(sensor_configs["camera"]["pos_ned"], dtype=float)

    # -------------------------
    # Ground truth
    # -------------------------
    for target_index, target_id in enumerate(
        sorted(scenario_data["ground_truth"], key=int)
    ):
        _, north, east = truth_track(
            scenario_data, target_id=int(target_id)
        )

        color = RUN_COLORS[target_index % len(RUN_COLORS)]

        ax.plot(
            east, north,
            "--",
            color=color,
            lw=1.6,
            alpha=0.65,
            label=f"truth {target_id}",
        )
        ax.plot(east[0], north[0], "o", color=color, ms=5)
        ax.plot(east[-1], north[-1], "D", color=color, ms=5)

    # -------------------------
    # Radar
    # -------------------------
    radar_range = sensor_configs["radar"]["range_m"]

    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            radar_range,
            fill=False,
            color=SENSOR_COLORS["radar"],
            ls="--",
            lw=1.0,
            alpha=0.25,
        )
    )
    ax.plot(
        0.0, 0.0,
        "^",
        color=SENSOR_COLORS["radar"],
        ms=9,
        label="Radar",
    )

    # -------------------------
    # Camera
    # -------------------------
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
    ax.plot(
        camera_pos[1],
        camera_pos[0],
        "s",
        color=SENSOR_COLORS["camera"],
        ms=8,
        label="Camera",
    )

    # -------------------------
    # Tracks (same logic as T6)
    # -------------------------
    best_track_ids = set(target_track_ids.values())

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
            continue  # safety

        color = RUN_COLORS[int(target_id) % len(RUN_COLORS)]

        ax.plot(
            history[:, 1],
            history[:, 0],
            "-",
            color=color,
            lw=2.2,
            label=f"track {track.track_id} -> target {target_id}",
        )

    # -------------------------
    # Formatting
    # -------------------------
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2)

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

    plot_multi_target_scene(ax, scenario_data, tracks, target_track_ids)

    ax.set_title(title)
    
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_t7_tracks(scenario_data, tracks, target_track_IDs, MOTP_time_series, CE_time_series, output_name, title):
    TRACKING_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TRACKING_OUTPUT_DIR / output_name

    sensor_configs = scenario_data["sensor_configs"]
    camera_pos = np.array(sensor_configs["camera"]["pos_ned"], dtype=float)

    fig, (ax_scene, ax_motp, ax_ce) = plt.subplots(1, 3, figsize=(18, 6))

    # -------------------------
    # 1. SCENE (same as T6)
    # -------------------------
    plot_multi_target_scene(ax_scene, scenario_data, tracks, target_track_IDs)

    ax_scene.set_title("Tracking Scene")

    # -------------------------
    # 2. MOTP plot
    # -------------------------
    times_motp = [t for t, _ in MOTP_time_series]
    motp_values = [v for _, v in MOTP_time_series]

    ax_motp.plot(times_motp, motp_values, color="#D97706", lw=2)
    ax_motp.set_title("MOTP (Localization Error)")
    ax_motp.set_xlabel("Time [s]")
    ax_motp.set_ylabel("Error [m]")
    ax_motp.grid(True, alpha=0.25)

    # -------------------------
    # 3. CE plot
    # -------------------------
    times_ce = [t for t, _ in CE_time_series]
    ce_values = [v for _, v in CE_time_series]

    ax_ce.plot(times_ce, ce_values, color="#DC2626", lw=2)

    ax_ce.set_title("Cardinality Error (CE)")
    ax_ce.set_xlabel("Time [s]")
    ax_ce.set_ylabel("|Tracks - Truth|")
    ax_ce.grid(True, alpha=0.25)

    # -------------------------
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Real data plots
# ─────────────────────────────────────────────────────────────────────────────

def _ned_to_webmercator(north_m, east_m, origin_lat, origin_lon):
    """Convert NED offsets to Web Mercator (EPSG:3857) metres."""
    R = 6_378_137.0
    lat = origin_lat + north_m / 111_111.0
    lon = origin_lon + east_m  / (111_111.0 * np.cos(np.deg2rad(origin_lat)))
    x   = np.deg2rad(lon) * R
    y   = np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0)) * R
    return x, y


def _track_span(track):
    """Return the maximum displacement [m] from the first position in a track."""
    if len(track.history) < 2:
        return 0.0
    hist = np.array([pos for _, pos in track.history])
    return float(np.max(np.linalg.norm(hist - hist[0], axis=1)))


# Distinct colours for the first N matched tracks
_MATCH_COLORS = ["#E11D48", "#D97706", "#7C3AED", "#059669", "#0284C7", "#DC2626"]


def plot_real_data_satellite(
    confirmed_tracks,
    gnss_measurements,
    ais_measurements,
    rmse_assignments,
    output_name,
    ned_origin_lat,
    ned_origin_lon,
    moving_threshold_m=60.0,
    coverage_radius_m=1600.0,
):
    """
    Save a trajectory plot overlaid on an OSM tile (or plain NED grid).

    The GNSS ground-truth track is shown **in full** (no clipping) so the
    complete vessel departure route is visible.  The axis bounds are set to
    the GNSS bounding box.  AIS reference ships are clipped to the GNSS
    bounding box to avoid the fan-spoke pattern from distant vessels.

    Tracks are split into three visual classes:
      • Matched  — confirmed track paired with a GNSS/AIS reference;
                   drawn in a distinct colour with RMSE in the legend.
      • Moving   — span > *moving_threshold_m* but no reference match;
                   drawn in semi-transparent dark grey.
      • Static   — span ≤ threshold (harbour clutter / fixed reflectors);
                   drawn as very faint dots, excluded from the legend.
    """
    TRACKING_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TRACKING_OUTPUT_DIR / output_name

    fig, ax = plt.subplots(figsize=(12, 11))

    def to_plot(n, e):
        if _HAS_CONTEXTILY:
            return _ned_to_webmercator(n, e, ned_origin_lat, ned_origin_lon)
        return e, n

    # ── Full GNSS ground truth ────────────────────────────────────────
    gnss_sorted = sorted(gnss_measurements, key=lambda m: m["time"])
    gx = [to_plot(m["north_m"], m["east_m"])[0] for m in gnss_sorted]
    gy = [to_plot(m["north_m"], m["east_m"])[1] for m in gnss_sorted]
    ax.plot(gx, gy, color=SENSOR_COLORS["gnss"], lw=2.0, alpha=0.85,
            label="GNSS ground truth (vessel)", zorder=5)
    ax.plot(gx[0], gy[0], "o", color=SENSOR_COLORS["gnss"], ms=9, zorder=6,
            label="Vessel start")
    ax.plot(gx[-1], gy[-1], "D", color=SENSOR_COLORS["gnss"], ms=8, zorder=6,
            label="Vessel end")

    # Derive axis bounds from the GNSS track extent with 5 % padding
    pad_x = (max(gx) - min(gx)) * 0.05 or 500
    pad_y = (max(gy) - min(gy)) * 0.05 or 500
    x_lo, x_hi = min(gx) - pad_x, max(gx) + pad_x
    y_lo, y_hi = min(gy) - pad_y, max(gy) + pad_y

    # ── AIS refs — grouped by MMSI (permanent vessel ID), clipped to bbox ──
    # Using mmsi avoids the fan-spike artefact that appears when ais_id is used:
    # ais_id is a locally reused integer that maps to different vessels over time.
    def in_gnss_bbox(n, e):
        px, py = to_plot(n, e)
        return x_lo <= px <= x_hi and y_lo <= py <= y_hi

    ais_by_mmsi = {}
    for m in ais_measurements:
        ais_by_mmsi.setdefault(str(m["mmsi"]), []).append(m)

    ais_ref_colors = ["#F59E0B", "#10B981", "#6366F1"]
    ais_plotted = 0
    for mmsi, msgs in sorted(ais_by_mmsi.items(),
                             key=lambda kv: min(np.sqrt(mm["north_m"]**2 + mm["east_m"]**2)
                                               for mm in kv[1])):
        msgs.sort(key=lambda m: m["time"])
        # Only keep messages that fall within the current view bbox
        clip = [m for m in msgs if in_gnss_bbox(m["north_m"], m["east_m"])]
        if len(clip) < 3:
            continue
        color = ais_ref_colors[ais_plotted % len(ais_ref_colors)]
        ax.plot(
            [to_plot(m["north_m"], m["east_m"])[0] for m in clip],
            [to_plot(m["north_m"], m["east_m"])[1] for m in clip],
            "--", color=color, lw=1.6, alpha=0.75,
            label=f"AIS ref – MMSI {mmsi} ({len(clip)} pts)", zorder=4,
        )
        ais_plotted += 1

    # ── Classify EKF tracks ───────────────────────────────────────────
    matched_ids = set(rmse_assignments.keys())
    moving_tracks, static_tracks = [], []
    for tr in confirmed_tracks:
        if tr.track_id in matched_ids:
            continue
        (moving_tracks if _track_span(tr) > moving_threshold_m else static_tracks).append(tr)

    # Static clutter — very faint dots, no legend entry
    for tr in static_tracks:
        if len(tr.history) < 2:
            continue
        hx = [to_plot(p[0], p[1])[0] for _, p in tr.history]
        hy = [to_plot(p[0], p[1])[1] for _, p in tr.history]
        ax.plot(hx, hy, ".", color="#9CA3AF", ms=1.5, alpha=0.15, zorder=1)

    # Moving unmatched — semi-transparent grey
    _moving_label_added = False
    for tr in moving_tracks:
        if len(tr.history) < 2:
            continue
        hx = [to_plot(p[0], p[1])[0] for _, p in tr.history]
        hy = [to_plot(p[0], p[1])[1] for _, p in tr.history]
        lbl = f"Moving (unmatched) tracks  n={len(moving_tracks)}" \
              if not _moving_label_added else "_nolegend_"
        ax.plot(hx, hy, "-", color="#475569", lw=1.0, alpha=0.40, zorder=2, label=lbl)
        ax.plot(hx[-1], hy[-1], ">", color="#475569", ms=4, alpha=0.40, zorder=2)
        _moving_label_added = True

    # Matched tracks — distinct colours, annotation arrow where track ends
    for idx, (tid, (ref_key, rmse_val)) in enumerate(sorted(rmse_assignments.items())):
        tr = next((t for t in confirmed_tracks if t.track_id == tid), None)
        if tr is None or len(tr.history) < 2:
            continue
        hx = [to_plot(p[0], p[1])[0] for _, p in tr.history]
        hy = [to_plot(p[0], p[1])[1] for _, p in tr.history]
        color = _MATCH_COLORS[idx % len(_MATCH_COLORS)]
        t_start = tr.history[0][0]
        t_end   = tr.history[-1][0]
        ax.plot(hx, hy, "-", color=color, lw=2.8, zorder=6,
                label=f"Track {tid} → {ref_key}  RMSE={rmse_val:.1f} m  "
                      f"(t={t_start:.0f}–{t_end:.0f} s)")
        ax.plot(hx[0], hy[0], "o", color=color, ms=8, zorder=7)
        ax.plot(hx[-1], hy[-1], "X", color=color, ms=10, zorder=7,
                label=f"Track {tid} lost at t={t_end:.0f} s  (vessel left sensor range)")

    # ── Sensor origin marker ──────────────────────────────────────────
    ox, oy = to_plot(0.0, 0.0)
    ax.plot(ox, oy, "^", color=SENSOR_COLORS["radar"], ms=12, zorder=8,
            label="Radar / Camera (NED origin)")

    # ── Apply axis limits BEFORE fetching tiles ───────────────────────
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # ── Map tiles ────────────────────────────────────────────────────
    if _HAS_CONTEXTILY:
        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:3857",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=14,
                attribution=False,
            )
            ax.set_xlabel("Easting (Web Mercator) [m]")
            ax.set_ylabel("Northing (Web Mercator) [m]")
        except Exception:
            ax.set_xlabel("East [m]")
            ax.set_ylabel("North [m]")
            ax.grid(True, alpha=0.3)
    else:
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    n_static  = len(static_tracks)
    n_moving  = len(moving_tracks)
    n_matched = len(rmse_assignments)
    ax.set_title(
        "Phase 4 — Confirmed tracks overlaid on harbour map\n"
        "(Copenhagen harbour, 5 March 2026 — full vessel departure route shown)\n"
        f"matched={n_matched}  moving={n_moving}  static/clutter={n_static}",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.88)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
    TRACKING_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TRACKING_OUTPUT_DIR / output_name

    fig, ax = plt.subplots(figsize=(11, 10))

    def to_plot(n, e):
        if _HAS_CONTEXTILY:
            return _ned_to_webmercator(n, e, ned_origin_lat, ned_origin_lon)
        return e, n

    def in_coverage(n, e):
        return np.sqrt(n ** 2 + e ** 2) <= coverage_radius_m

    # ── Classify tracks ───────────────────────────────────────────────
    matched_ids  = set(rmse_assignments.keys())
    moving_tracks, static_tracks = [], []
    for tr in confirmed_tracks:
        if tr.track_id in matched_ids:
            continue
        (moving_tracks if _track_span(tr) > moving_threshold_m else static_tracks).append(tr)

    # ── Static clutter (very faint, no legend entry) ──────────────────
    for tr in static_tracks:
        if len(tr.history) < 2:
            continue
        hx = [to_plot(p[0], p[1])[0] for _, p in tr.history]
        hy = [to_plot(p[0], p[1])[1] for _, p in tr.history]
        ax.plot(hx, hy, ".", color="#9CA3AF", ms=1.5, alpha=0.18, zorder=1)

    # ── Moving unmatched tracks (grey) ────────────────────────────────
    _moving_label_added = False
    for tr in moving_tracks:
        if len(tr.history) < 2:
            continue
        hx = [to_plot(p[0], p[1])[0] for _, p in tr.history]
        hy = [to_plot(p[0], p[1])[1] for _, p in tr.history]
        lbl = f"Moving tracks (n={len(moving_tracks)})" if not _moving_label_added else "_nolegend_"
        ax.plot(hx, hy, "-", color="#475569", lw=1.0, alpha=0.45, zorder=2, label=lbl)
        ax.plot(hx[-1], hy[-1], ">", color="#475569", ms=4, alpha=0.45, zorder=2)
        _moving_label_added = True

    # ── Matched tracks (distinct colours) ────────────────────────────
    for idx, (tid, (ref_key, rmse_val)) in enumerate(sorted(rmse_assignments.items())):
        tr = next((t for t in confirmed_tracks if t.track_id == tid), None)
        if tr is None or len(tr.history) < 2:
            continue
        hx = [to_plot(p[0], p[1])[0] for _, p in tr.history]
        hy = [to_plot(p[0], p[1])[1] for _, p in tr.history]
        color = _MATCH_COLORS[idx % len(_MATCH_COLORS)]
        ax.plot(hx, hy, "-", color=color, lw=2.5, zorder=6,
                label=f"Track {tid} → {ref_key}  (RMSE={rmse_val:.1f} m)")
        ax.plot(hx[0], hy[0], "o", color=color, ms=7, zorder=7)
        ax.plot(hx[-1], hy[-1], ">", color=color, ms=8, zorder=7)

    # ── GNSS ground-truth — clipped to coverage window ───────────────
    gnss_sorted = sorted(gnss_measurements, key=lambda m: m["time"])
    gnss_clip   = [m for m in gnss_sorted if in_coverage(m["north_m"], m["east_m"])]
    if gnss_clip:
        gx = [to_plot(m["north_m"], m["east_m"])[0] for m in gnss_clip]
        gy = [to_plot(m["north_m"], m["east_m"])[1] for m in gnss_clip]
        ax.plot(gx, gy, color=SENSOR_COLORS["gnss"], lw=2.0, alpha=0.85,
                label="GNSS ground truth (vessel)", zorder=5)
        ax.plot(gx[0], gy[0], "o", color=SENSOR_COLORS["gnss"], ms=8, zorder=5)
        ax.plot(gx[-1], gy[-1], "D", color=SENSOR_COLORS["gnss"], ms=7, zorder=5)

    # ── AIS reference — only ships with at least one fix inside coverage ──
    ais_by_id = {}
    for m in ais_measurements:
        ais_by_id.setdefault(str(m["ais_id"]), []).append(m)

    ais_ref_colors = ["#F59E0B", "#10B981", "#6366F1"]
    ais_plotted = 0
    for aid, msgs in sorted(ais_by_id.items()):
        msgs.sort(key=lambda m: m["time"])
        # Keep only the portion of the track that is within the coverage window
        clip = [m for m in msgs if in_coverage(m["north_m"], m["east_m"])]
        if len(clip) < 3:
            continue
        color = ais_ref_colors[ais_plotted % len(ais_ref_colors)]
        ax.plot(
            [to_plot(m["north_m"], m["east_m"])[0] for m in clip],
            [to_plot(m["north_m"], m["east_m"])[1] for m in clip],
            "--", color=color, lw=1.4, alpha=0.75,
            label=f"AIS ref – ship {aid} ({len(clip)} pts)", zorder=4,
        )
        ais_plotted += 1

    # ── Sensor origin marker ──────────────────────────────────────────
    ox, oy = to_plot(0.0, 0.0)
    ax.plot(ox, oy, "^", color=SENSOR_COLORS["radar"], ms=12, zorder=8,
            label="Radar / Camera (NED origin)")

    # ── Set axis limits BEFORE adding tiles ──────────────────────────
    lim_lo_x, lim_lo_y = to_plot(-coverage_radius_m, -coverage_radius_m)
    lim_hi_x, lim_hi_y = to_plot( coverage_radius_m,  coverage_radius_m)
    ax.set_xlim(min(lim_lo_x, lim_hi_x), max(lim_lo_x, lim_hi_x))
    ax.set_ylim(min(lim_lo_y, lim_hi_y), max(lim_lo_y, lim_hi_y))

    # ── Map tiles ────────────────────────────────────────────────────
    if _HAS_CONTEXTILY:
        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:3857",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=15,
                attribution=False,
            )
            ax.set_xlabel("Easting (Web Mercator) [m]")
            ax.set_ylabel("Northing (Web Mercator) [m]")
        except Exception:
            ax.set_xlabel("East [m]")
            ax.set_ylabel("North [m]")
            ax.grid(True, alpha=0.3)
    else:
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    n_static  = len(static_tracks)
    n_moving  = len(moving_tracks)
    n_matched = len(rmse_assignments)
    ax.set_title(
        "Phase 4 — Real-data tracks overlaid on harbour map\n"
        f"(Copenhagen harbour, 5 March 2026 — {coverage_radius_m:.0f} m coverage window)   "
        f"matched={n_matched}  moving={n_moving}  static/clutter={n_static}",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_real_data_metrics(
    motp_real,
    motp_sim,
    ce_real,
    ce_sim,
    rmse_assignments,
    output_name,
):
    """
    Five-panel figure comparing real-data and simulation metrics.

    MOTP and CE are shown in separate sub-rows (real / simulation) so that
    the very different time ranges (real: 200–1200 s; sim: 0–180 s) do not
    compress one curve to a thin sliver.  Scalar averages are annotated on
    each panel.  The right column shows the per-track RMSE bar chart.
    """
    TRACKING_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TRACKING_OUTPUT_DIR / output_name

    # Layout: 2 rows × 3 cols; RMSE spans both rows on the right
    fig = plt.figure(figsize=(18, 8))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    ax_motp_real = fig.add_subplot(gs[0, 0])
    ax_motp_sim  = fig.add_subplot(gs[1, 0])
    ax_ce_real   = fig.add_subplot(gs[0, 1])
    ax_ce_sim    = fig.add_subplot(gs[1, 1])
    ax_rmse      = fig.add_subplot(gs[:, 2])   # spans both rows

    def _plot_series(ax, series, color, title, ylabel, avg=None):
        if series:
            t, v = zip(*series)
            # Clip extreme outliers (> 99th percentile) so y-axis stays readable
            v_arr   = np.array(v)
            p99     = float(np.percentile(v_arr, 99))
            v_clip  = np.clip(v_arr, 0, p99)
            ax.plot(t, v_clip, color=color, lw=1.6)
            if avg is not None and np.isfinite(avg):
                ax.axhline(avg, color=color, lw=1.2, ls="--", alpha=0.6,
                           label=f"mean = {avg:.1f}")
                ax.legend(fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)

    # MOTP — real
    motp_real_avg = float(np.mean([v for _, v in motp_real])) if motp_real else float("nan")
    _plot_series(ax_motp_real, motp_real, "#E11D48",
                 "MOTP — Real data (Copenhagen harbour)", "MOTP [m]", motp_real_avg)

    # MOTP — simulation (filter out scans where motp > 200 m, which are
    # false-alarm tracks dominating the average before the scene stabilises)
    motp_sim_filt = [(t, v) for t, v in (motp_sim or []) if v <= 200.0]
    motp_sim_avg  = float(np.mean([v for _, v in motp_sim_filt])) if motp_sim_filt else float("nan")
    _plot_series(ax_motp_sim, motp_sim_filt, "#2E75B6",
                 "MOTP — Simulation Scenario E", "MOTP [m]", motp_sim_avg)

    # CE — real
    ce_real_avg = float(np.mean([v for _, v in ce_real])) if ce_real else float("nan")
    _plot_series(ax_ce_real, ce_real, "#E11D48",
                 "CE — Real data", "CE (|N̂ − N|)", ce_real_avg)

    # CE — simulation (filter CE > 10 for same reason)
    ce_sim_filt = [(t, v) for t, v in (ce_sim or []) if v <= 10.0]
    ce_sim_avg  = float(np.mean([v for _, v in ce_sim_filt])) if ce_sim_filt else float("nan")
    _plot_series(ax_ce_sim, ce_sim_filt, "#2E75B6",
                 "CE — Simulation Sc-E  (excess = false-alarm tracks)", "CE (|N̂ − N|)", ce_sim_avg)

    # Summary table inside MOTP-sim panel
    ax_motp_sim.text(
        0.99, 0.97,
        f"Real  MOTP avg : {motp_real_avg:.1f} m\n"
        f"Sim   MOTP avg : {motp_sim_avg:.1f} m\n"
        f"Real  CE avg   : {ce_real_avg:.2f}\n"
        f"Sim   CE avg   : {ce_sim_avg:.2f}",
        transform=ax_motp_sim.transAxes,
        va="top", ha="right", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # RMSE bar chart
    if rmse_assignments:
        labels = [f"T{tid}\n→ {rk}" for tid, (rk, _) in sorted(rmse_assignments.items())]
        values = [rv for _, (_, rv) in sorted(rmse_assignments.items())]
        colors = ["#E11D48", "#D97706", "#7C3AED", "#2E75B6"]
        bars   = ax_rmse.bar(labels, values,
                             color=colors[:len(labels)],
                             edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, values):
            ax_rmse.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f"{val:.1f} m", ha="center", va="bottom", fontsize=10,
                         fontweight="bold")
        # Add simulation reference line
        if np.isfinite(motp_sim_avg):
            ax_rmse.axhline(motp_sim_avg, color="#2E75B6", lw=1.5, ls="--",
                            label=f"Sim MOTP avg ({motp_sim_avg:.1f} m)")
            ax_rmse.legend(fontsize=8)
    else:
        ax_rmse.text(0.5, 0.5, "No tracks matched\nto ground truth",
                     ha="center", va="center", transform=ax_rmse.transAxes)
    ax_rmse.set_ylabel("RMSE [m]", fontsize=9)
    ax_rmse.set_title("Per-track RMSE vs ground truth\n(real data only)", fontsize=9)
    ax_rmse.grid(True, alpha=0.25, axis="y")
    ax_rmse.tick_params(labelsize=9)

    fig.suptitle(
        "Phase 4 — Quantitative accuracy metrics  (real data vs simulation Scenario E)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sim_vs_real_comparison(
    real_confirmed,
    real_gnss,
    real_ais,
    sim_data,
    sim_tracks,
    output_name,
    rmse_assignments=None,
    view_radius_m=1600.0,
):
    """
    Side-by-side NED scene comparison:
      Left  : real data — sensor coverage window (view_radius_m), GNSS clipped
               to that window, AIS grouped by MMSI (not reused ais_id).
      Right : simulation Scenario E tracks + ground truth.
    """
    TRACKING_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TRACKING_OUTPUT_DIR / output_name

    fig, (ax_real, ax_sim) = plt.subplots(1, 2, figsize=(16, 8))

    # ── Left: real data ───────────────────────────────────────────────
    # Clip GNSS to the view window so the axis stays readable
    gnss_s = sorted(real_gnss, key=lambda m: m["time"])
    gnss_clip = [m for m in gnss_s
                 if np.sqrt(m["north_m"]**2 + m["east_m"]**2) <= view_radius_m]
    if gnss_clip:
        ax_real.plot(
            [m["east_m"]  for m in gnss_clip],
            [m["north_m"] for m in gnss_clip],
            color=SENSOR_COLORS["gnss"], lw=2.0, alpha=0.85,
            label="GNSS ground truth (vessel)",
        )
        ax_real.plot(gnss_clip[0]["east_m"],  gnss_clip[0]["north_m"],
                     "o", color=SENSOR_COLORS["gnss"], ms=8)
        ax_real.plot(gnss_clip[-1]["east_m"], gnss_clip[-1]["north_m"],
                     "D", color=SENSOR_COLORS["gnss"], ms=7)

    # AIS — group by MMSI, clip to view window
    ais_by_mmsi = {}
    for m in real_ais:
        ais_by_mmsi.setdefault(str(m["mmsi"]), []).append(m)
    ais_colors = ["#F59E0B", "#10B981", "#6366F1"]
    ais_plotted = 0
    for mmsi, msgs in sorted(ais_by_mmsi.items()):
        msgs.sort(key=lambda m: m["time"])
        clip = [m for m in msgs
                if np.sqrt(m["north_m"]**2 + m["east_m"]**2) <= view_radius_m]
        if len(clip) < 3:
            continue
        color = ais_colors[ais_plotted % len(ais_colors)]
        ax_real.plot(
            [m["east_m"]  for m in clip],
            [m["north_m"] for m in clip],
            "--", color=color, lw=1.4, alpha=0.70,
            label=f"AIS MMSI {mmsi}",
        )
        ais_plotted += 1

    # Tracks — matched in colour, static clutter as faint dots; unmatched
    # moving tracks are hidden (they are mostly spurious tracks formed by the
    # tracker jumping between clutter returns and add no interpretable info).
    matched_ids = set((rmse_assignments or {}).keys())
    for track in real_confirmed:
        if len(track.history) < 2:
            continue
        hist = np.array([pos for _, pos in track.history])
        span = float(np.max(np.linalg.norm(hist - hist[0], axis=1)))
        if track.track_id in matched_ids:
            ref_key, rmse_val = rmse_assignments[track.track_id]
            color = _MATCH_COLORS[sorted(matched_ids).index(track.track_id) % len(_MATCH_COLORS)]
            ax_real.plot(hist[:, 1], hist[:, 0], "-", color=color, lw=2.5, zorder=5,
                         label=f"Track {track.track_id}→{ref_key} RMSE={rmse_val:.1f}m")
            ax_real.plot(hist[-1, 1], hist[-1, 0], "X", color=color, ms=9, zorder=6)
        elif span <= 60.0:
            # Static clutter — faint dots only
            ax_real.plot(hist[:, 1], hist[:, 0], ".", color="#9CA3AF",
                         ms=1.5, alpha=0.15, zorder=1)

    ax_real.plot(0.0, 0.0, "^", color=SENSOR_COLORS["radar"], ms=11, zorder=7,
                 label="Radar / Camera")
    ax_real.set_xlim(-view_radius_m, view_radius_m)
    ax_real.set_ylim(-view_radius_m, view_radius_m)
    ax_real.set_xlabel("East [m]")
    ax_real.set_ylabel("North [m]")
    ax_real.set_title("Real data — Copenhagen harbour\n"
                      f"(sensor coverage window ±{view_radius_m:.0f} m)")
    ax_real.set_aspect("equal", adjustable="box")
    ax_real.grid(True, alpha=0.25)
    ax_real.legend(fontsize=8, loc="upper right", framealpha=0.85)

    # ── Right: simulation ─────────────────────────────────────────────
    # For each target, keep the confirmed track with the most history points
    # (not just the last one seen during iteration, which is often a short
    # spurious track formed near the end of the scenario).
    from src.t6_gating_association import dominant_target_id
    sim_target_track_ids = {}   # {target_id -> track_id of best/longest track}
    sim_target_best_len  = {}
    for tr in sim_tracks:
        if not tr.confirmed:
            continue
        tid = dominant_target_id(tr)
        if tid is None:
            continue
        if len(tr.history) > sim_target_best_len.get(tid, 0):
            sim_target_best_len[tid]  = len(tr.history)
            sim_target_track_ids[tid] = tr.track_id

    plot_multi_target_scene(ax_sim, sim_data, sim_tracks, sim_target_track_ids)
    ax_sim.set_title(
        "Simulation — Scenario E  (synthetic scene)\n"
        "NED geometry ≠ Copenhagen harbour — comparison is on RMSE/MOTP numbers only"
    )

    fig.suptitle(
        "Phase 4 — Real data vs Simulation NED scene comparison",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path