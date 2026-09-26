import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[2] / "examples").glob("*/*.py"))


@pytest.mark.parametrize(
    "example", EXAMPLES, ids=[f"{p.parent.name}/{p.name}" for p in EXAMPLES]
)
def test_the_example_runs(example: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=example.parents[2],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert result.stdout.strip()
