import subprocess
import sys
from importlib.resources import as_file, files


WORKFLOWS = {
    "register": "registration.smk",
    "apply": "apply_transform.smk",
    "optimize-cpd": "cpd_optimization.smk",
}


def run_snakemake(workflow, configfile, cores=8, work_dir=".", unlock=False,
                    rerun_incomplete=False,):
    if workflow not in WORKFLOWS:
        raise ValueError(f"Unknown workflow: {workflow}")

    resource = files("image_matchmaker").joinpath(
        "workflows", WORKFLOWS[workflow]
    )

    with as_file(resource) as snakefile:
        cmd = [
            sys.executable, "-m", "snakemake",
            "--snakefile", str(snakefile),
            "--configfile", str(configfile),
            "--directory", str(work_dir),
            "--cores", str(cores),
        ]
        if unlock:
            cmd.append("--unlock")
        if rerun_incomplete:
            cmd.append("--rerun-incomplete")

        subprocess.run(cmd, check=True)
