# Quickstart

Run full registration workflow with Snakemake:

```bash
snakemake -s workflows/registration.smk \
    --configfile examples/register_config_test_rigid.yaml \
    --cores 8
```

Run tests:

```bash
pytest -s
```

Optional (hide warnings):

```bash
pytest -s -p no:warnings
```
