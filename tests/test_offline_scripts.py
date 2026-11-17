"""Run each data folder's self-contained offline check as a subprocess.

They are kept next to the builders they test (data/<x>/test_offline.py) so a
folder stays usable on its own; this file just makes pytest collect them.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "data" / "ecoli" / "test_offline.py",
    ROOT / "data" / "kegg" / "test_offline.py",
    ROOT / "data" / "gem" / "test_offline.py",
    ROOT / "experiments" / "splits" / "test_offline.py",
]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_offline_script(script):
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       cwd=str(script.parent), env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})
    assert r.returncode == 0, f"{script}\n--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}"
