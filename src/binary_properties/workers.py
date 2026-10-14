"""Launch generated-code workers with a filtered environment and a deadline."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def worker_environment(*, extra_path: Path | None = None) -> dict[str, str]:
    # An allowlist avoids inheriting API credentials under any provider's key name.
    env = {
        key: os.environ[key]
        for key in ("SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")
        if key in os.environ
    }
    paths = [str(Path(__file__).resolve().parents[1])]
    if extra_path is not None:
        paths.insert(0, str(extra_path))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_worker(
    module: str,
    payload: str,
    *,
    timeout: float,
    args: tuple[str, ...] = (),
    extra_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", module, *args]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=worker_environment(extra_path=extra_path),
    )
    try:
        stdout, stderr = process.communicate(
            payload, timeout=max(0.001, timeout - (time.monotonic() - started))
        )
    except BaseException:
        process.kill()
        process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
