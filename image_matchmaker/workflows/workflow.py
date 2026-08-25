import subprocess
import sys
from importlib.resources import as_file, files


WORKFLOWS = {
    "registration": "registration.smk",
    "apply-transform": "apply_transform.smk",
    "cpd-optimization": "cpd_optimization.smk"
}


def run_snakemake(workflow, configfile, snakefile=None, cores=8, work_dir=".",):
    if workflow == "custom":
        if snakefile is None:
            raise ValueError("--snakefile is required for custom workflow")

        _run_snakemake(snakefile, configfile, cores, work_dir)
        return
    elif workflow in WORKFLOWS:
        resource = files("image_matchmaker").joinpath(
            "workflows", WORKFLOWS[workflow]
        )

        with as_file(resource) as snakefile_path:
            _run_snakemake(snakefile_path, configfile, cores, work_dir)
    else:
        raise NotImplementedError(f"Unsupported workflow: {workflow}")


def _run_snakemake(snakefile, configfile, cores, work_dir):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile", str(snakefile),
            "--configfile", str(configfile),
            "--directory", str(work_dir),
            "--cores", str(cores),
        ],
        check=True,
    )
