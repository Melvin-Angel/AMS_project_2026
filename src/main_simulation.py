import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.t2_coordinate_manager import run_t2_coordinate_manager_tests
from src.t3_radar_tracker import run_t3_radar_tracker
from src.t4_camera_fusion import run_t4_comparison
from src.t5_ais_fusion import run_t5_comparison
from src.t6_gating_association import run_t6_association
from src.t7_track_management import run_t7_track_management


def main():
    parser = argparse.ArgumentParser(description="Run harbour surveillance EKF tasks.")
    parser.add_argument(
        "task",
        choices=("t2", "t3", "t4", "t5", "t6", "t7", "all"),
        nargs="?",
        default="t7",
        help="Task runner to execute. Defaults to all.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable tracking plot generation.",
    )
    args = parser.parse_args()

    make_plot = not args.no_plot

    if args.task in ("t2", "all"):
        run_t2_coordinate_manager_tests()

    if args.task in ("t3", "all"):
        run_t3_radar_tracker(make_plot=make_plot)

    if args.task in ("t4", "all"):
        run_t4_comparison(make_plot=make_plot)

    if args.task in ("t5", "all"):
        run_t5_comparison(make_plot=make_plot)

    if args.task in ("t6", "all"):
        run_t6_association(make_plot=make_plot)

    if args.task in ("t7", "all"):
        run_t7_track_management(make_plot=make_plot)


if __name__ == "__main__":
    main()
