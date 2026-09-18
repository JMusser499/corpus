"""SLURM launch controls needed by corpuscles with distinct environments.

The phase scripts are shared across corpuscles.  A launcher must be able to
select the reviewed conda environment, and an OCR-panel corpus must be able to
omit the separate vision-GPU phase without omitting embedding or finalize.
"""
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parent.parent
SLURM = REPO / "slurm"


def test_phase_jobs_activate_the_selected_conda_environment():
    paths = (SLURM / "bouchet_paths.sh").read_text()
    assert 'CORPUS_CONDA_ENV="${CORPUS_CONDA_ENV:-corpus}"' in paths
    assert "export CORPUS_CONDA_ENV" in paths

    for name in (
        "batch_process_corpus.sh",
        "batch_pass3b.sh",
        "batch_embed.sh",
        "batch_finalize.sh",
    ):
        script = (SLURM / name).read_text()
        assert 'conda activate "$CORPUS_CONDA_ENV"' in script, name


def test_launcher_can_omit_vision_without_omitting_embed_or_finalize():
    script = (SLURM / "batch_pipeline.sh").read_text()

    assert 'RUN_VISION="${RUN_VISION:-1}"' in script
    assert 'if [ "$RUN_VISION" = "1" ]' in script
    assert 'RUN_VISION must be 0 or 1' in script
    assert 'Submitting Embed (GPU)' in script
    assert 'Submitting Finalize (cross-paper tail)' in script
    assert 'FINALIZE_DEPENDENCY="afterok:$EMBED_JOB"' in script


def test_runbook_documents_environment_and_vision_controls():
    runbook = (REPO / "dev_docs" / "BOUCHET.md").read_text()

    assert "CORPUS_CONDA_ENV" in runbook
    assert "RUN_VISION" in runbook


def test_no_vision_launch_builds_a_valid_dependency_chain(tmp_path):
    """Exercise the launcher with scheduler commands replaced by recorders."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    counter = tmp_path / "counter"
    counter.write_text("100\n")

    commands = {
        "sbatch": f"""#!/bin/bash
set -euo pipefail
id=$(($(cat {counter}) + 1))
echo "$id" > {counter}
printf '%s\\n' "$*" >> {calls}
echo "$id"
""",
        "squeue": """#!/bin/bash
case "$*" in
  *%T*) echo RUNNING ;;
  *%N*) echo mocknode ;;
  *) echo mockqueue ;;
esac
""",
        "curl": "#!/bin/bash\nexit 0\n",
        "corpus": "#!/bin/bash\nexit 0\n",
    }
    for name, body in commands.items():
        path = fake_bin / name
        path.write_text(body)
        os.chmod(path, 0o755)

    inputs = tmp_path / "pdfs"
    inputs.mkdir()
    (inputs / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"input_pdfs: {inputs}\n"
        f"output_dir: {tmp_path / 'output'}\n"
        "grobid:\n  disable: true\n"
    )

    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "BOUCHET_PROJECT": str(tmp_path),
        "CACHE_DIR": str(tmp_path / "cache"),
        "CORPUS_CONFIG": str(config),
        "CORPUS_CONDA_ENV": "test-env",
        "REPO_DIR": str(REPO),
        "RUN_VISION": "0",
    })
    result = subprocess.run(
        ["bash", str(SLURM / "batch_pipeline.sh")],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    submitted = calls.read_text()
    assert "batch_pass3b.sh" not in submitted
    assert "batch_embed.sh" in submitted
    assert "batch_finalize.sh" in submitted
    assert "--dependency=afterok:104" in submitted
    assert "Pass 3b:             disabled (RUN_VISION=0)" in result.stdout
