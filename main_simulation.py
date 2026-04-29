import argparse

from t2_coordinate_manager import run_t2_coordinate_manager_tests
from t3_radar_tracker import run_t3_radar_tracker
from t4_camera_fusion import run_t4_comparison
from t5_ais_fusion import run_t5_comparison


def main():
    parser = argparse.ArgumentParser(description="Run harbour surveillance EKF tasks.")
    parser.add_argument(
        "task",
        choices=("t2", "t3", "t4", "t5", "all"),
        nargs="?",
        default="t5",
        help="Task runner to execute. Defaults to t5.",
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


if __name__ == "__main__":
    main()
