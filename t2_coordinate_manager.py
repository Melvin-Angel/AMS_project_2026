import unittest


def run_t2_coordinate_manager_tests():
    print("Starting T2 coordinate manager unit tests...", flush=True)
    suite = unittest.defaultTestLoader.discover("tests", pattern="test_coordinate_manager.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run_t2_coordinate_manager_tests()
