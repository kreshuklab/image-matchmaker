import pytest

from test_raw_to_n5 import get_test_path, test_raw_to_n5
from test_prealignment import test_prealignment
from test_rigid_alignment import test_rigid_alignment
from test_compare_results import test_compare_results


def test_full_pipeline():
    try:
        for input_path, output_path in get_test_path():
            test_raw_to_n5(input_path, output_path)
        print("\n✅ raw_to_n5 finished.")
    except Exception as e:
        pytest.fail(f"raw_to_n5 failed: {e}")

    try:
        test_prealignment()
        print("✅ prealignment finished.")
    except Exception as e:
        pytest.fail(f"prealignment failed: {e}")

    try:
        test_rigid_alignment()
        print("✅ rigid_alignment finished.")
    except Exception as e:
        pytest.fail(f"rigid_alignment failed: {e}")

    try:
        test_compare_results()
        print("✅ compare_results finished.")
    except Exception as e:
        pytest.fail(f"compare_results failed: {e}")

    print("✅ Full pipeline completed successfully.")
