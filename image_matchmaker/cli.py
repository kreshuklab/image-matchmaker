import click

from image_matchmaker.workflows import run_snakemake


def workflow_options(func):
    func = click.option("--config", "configfile", type=click.Path(exists=True),
                        required=True,)(func)
    func = click.option("--cores", default=8, type=int, show_default=True,)(func)
    func = click.option("--work-dir", default=".", type=click.Path(file_okay=False),
                        show_default=True,)(func)

    return func


@click.group()
def main():
    """Run image-matchmaker workflows."""


@main.command()
@workflow_options
def register(configfile, cores, work_dir):
    """Run registration."""
    run_snakemake("register", configfile, cores, work_dir)


@main.command()
@workflow_options
def apply(configfile, cores, work_dir):
    """Apply a registration transform."""
    run_snakemake("apply", configfile, cores, work_dir)


@main.command()
@workflow_options
def optimize_cpd(configfile, cores, work_dir):
    """Run CPD optimization."""
    run_snakemake("optimize-cpd", configfile, cores, work_dir)


if __name__ == "__main__":
    main()
