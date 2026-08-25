import click

from image_matchmaker.workflows import run_snakemake


@click.command()
@click.option("--workflow", default="registration", show_default=True,
                type=click.Choice(["registration", "apply-transform", "cpd-optimization","custom"]),)
@click.option("--snakefile", type=click.Path(exists=True), default=None)
@click.option("--config", "configfile", type=click.Path(exists=True), required=True,)
@click.option("--cores", default=8, type=int, show_default=True)
@click.option("--work-dir", default=".", type=click.Path(file_okay=False), show_default=True,)
def main(workflow, snakefile, configfile, cores, work_dir):
    run_snakemake(
        workflow=workflow,
        configfile=configfile,
        snakefile=snakefile,
        cores=cores,
        work_dir=work_dir,
    )

if __name__ == "__main__":
    main()
