import os
from click.testing import CliRunner

from matchmaker.align_rigid_elastix import main
from matchmaker.utils import load_test_config, get_n5_path, read_volume


def test_rigid_alignment():
    config = load_test_config()
    test_dir = config["log_dir"]
    assert os.path.exists(test_dir)

    input_key = config["keys"]["prealignment"]
    output_key = config["keys"]["rigid_alignment"]

    output_dir = f"{test_dir}/{output_key}"
    os.makedirs(output_dir, exist_ok=True)

    fixed_path, moving_path = get_n5_path()

    runner = CliRunner()

    runner.invoke(main, [
        '--fixed_path', fixed_path,
        '--fixed_key', input_key,
        '--moving_path', moving_path,
        '--moving_key', input_key,
        '--output_dir', output_dir,
        '--output_key', output_key,
    ])

    read_volume(moving_path, output_key)
