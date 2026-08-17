import importlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "image_matchmaker"
SCRIPT_PATTERN = re.compile(rf"{PACKAGE}/[\w/]+\.py")


def _module_name(path):
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def _import_failures(paths):
    failures = []
    for path in sorted(set(paths)):
        try:
            importlib.import_module(_module_name(path))
        except Exception as e:
            failures.append(f"{_module_name(path)}: {type(e).__name__}: {e}")
    return failures


def test_every_module_imports():
    # walk files, not pkgutil.walk_packages: cpd_parameter_tuning/ has no __init__.py, so
    # walk_packages skips it without reporting anything
    modules = [p for p in (REPO_ROOT / PACKAGE).rglob("*.py") if "__pycache__" not in p.parts]
    assert modules, "found no modules to import"

    failures = _import_failures(modules)
    assert not failures, "modules failing to import:\n  " + "\n  ".join(failures)


def test_workflow_scripts_exist_and_import():
    snakefiles = sorted((REPO_ROOT / "workflows").glob("*.smk"))
    assert snakefiles, "found no workflow files in workflows/"

    scripts = {
        REPO_ROOT / match
        for snakefile in snakefiles
        for match in SCRIPT_PATTERN.findall(snakefile.read_text())
    }
    assert scripts, "found no script paths in the workflows"

    missing = sorted(str(p.relative_to(REPO_ROOT)) for p in scripts if not p.is_file())
    assert not missing, "workflow rules reference missing scripts:\n  " + "\n  ".join(missing)

    failures = _import_failures(scripts)
    assert not failures, "scripts called by the workflows fail to import:\n  " + "\n  ".join(failures)
