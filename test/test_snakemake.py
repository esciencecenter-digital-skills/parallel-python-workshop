from subprocess import run
from shutil import copy


def test_snakemake(tmp_path):
    copy("./test/hello/Snakefile", tmp_path)
    run(["snakemake", "-j1"], cwd=tmp_path, check=True)
    assert (tmp_path / "combined.txt").exists()
