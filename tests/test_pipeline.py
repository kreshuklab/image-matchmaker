import os
import subprocess

from examples.deform_test_data import deform_test_data
from tests.compare_results import compare_results


def run_pipline(config_path, snakefile, cores=8):
    deform_test_data(config_path)

    result = subprocess.run(
        [
            "snakemake",
            "--snakefile", snakefile,
            "--cores", str(cores),
        ],
        check=True,
    )
    assert result.returncode == 0, result.stderr

    compare_results(config_path)


def test_workflow():

    run_pipline("examples/register_config_test_rigid.yaml", "workflows/rigid_registration.smk")
