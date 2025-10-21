import os
import pytest
from click.testing import CliRunner

from matchmaker.raw_to_n5 import main
from matchmaker.utils import load_test_config, get_n5_path, read_volume


def get_test_path():
    config = load_test_config()

    fixed_input_path = config["fixed_image"]["path"]
    moving_input_path = config["moving_image"]["path"]

    fixed_output_path, moving_output_path = get_n5_path()

    return [(fixed_input_path, fixed_output_path), (moving_input_path, moving_output_path)]


@pytest.mark.parametrize("input_path, output_path", get_test_path())
def test_raw_to_n5(input_path, output_path):
    config = load_test_config()
    test_dir = config["log_dir"]
    os.makedirs(test_dir, exist_ok=True)

    output_key = config["keys"]["n5"]
    runner = CliRunner()

    runner.invoke(main, [
        '--input_path', input_path,
        '--output_path', output_path,
        '--output_key', output_key,
        '--log_dir', test_dir,
    ])

    read_volume(output_path, output_key)
