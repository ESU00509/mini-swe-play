import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

PREFIX = "longrun "


class LongJobEnvironmentConfig(LocalEnvironmentConfig):
    job_timeout: int = 3600
    """Seconds to wait for a `longrun` command before killing it. 0 means no limit."""
    log_tail_bytes: int = 6000
    """How much of the end of the job's log is returned to the agent."""


class LongJobEnvironment(LocalEnvironment):
    def __init__(self, *, config_class: type = LongJobEnvironmentConfig, **kwargs):
        """Local environment where commands starting with `longrun` can run far longer than `timeout`.
        The call blocks until the command ends, so the agent only continues (and analyzes the result) once it is done.
        """
        super().__init__(config_class=config_class, **kwargs)
        self.jobs_dir = Path(tempfile.mkdtemp(prefix="mini-longrun-"))
        self.n_jobs = 0

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        if not command.startswith(PREFIX):
            output = super().execute(action, cwd, timeout=timeout)
            if output.get("extra", {}).get("exception_type") == "TimeoutExpired":
                output["exception_info"] += "\nIf the command legitimately takes longer, run it as `longrun <command>`."
            return output
        self.n_jobs += 1
        log = self.jobs_dir / f"job{self.n_jobs}.log"
        start = time.monotonic()
        timed_out = False
        with log.open("wb") as f:
            process = subprocess.Popen(
                command.removeprefix(PREFIX),
                shell=True,
                cwd=cwd or self.config.cwd or os.getcwd(),
                env=os.environ | self.config.env,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            try:
                process.wait(timeout=self.config.job_timeout or None)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
                process.wait()
                timed_out = True
        size = log.stat().st_size
        with log.open("rb") as f:
            f.seek(max(0, size - self.config.log_tail_bytes))
            tail = f.read().decode("utf-8", errors="replace")
        status = (
            f"killed after exceeding job_timeout ({self.config.job_timeout}s)"
            if timed_out
            else f"exited with code {process.returncode}"
        )
        header = f"[longrun job {self.n_jobs} {status}, ran {time.monotonic() - start:.0f}s. Full log: {log}]\n"
        if size > self.config.log_tail_bytes:
            header += f"[showing only the last {self.config.log_tail_bytes} of {size} bytes]\n"
        return {"output": header + tail, "returncode": process.returncode, "exception_info": ""}
