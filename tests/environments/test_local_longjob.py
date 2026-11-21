import os
import re
from pathlib import Path

import pytest

from minisweagent.agents import get_agent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.environments.local_longjob import LongJobEnvironment, LongJobEnvironmentConfig
from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.utils.serialize import recursive_merge

pytestmark = pytest.mark.skipif(os.name != "posix", reason="uses POSIX shell commands")


def log_path(result: dict) -> Path:
    return Path(re.search(r"Full log: (.+)\]", result["output"]).group(1))


def test_longjob_config_defaults():
    assert (LongJobEnvironmentConfig().timeout, LongJobEnvironmentConfig().job_timeout) == (30, 3600)


def test_plain_command_still_times_out_and_hints_longrun():
    result = LongJobEnvironment(timeout=1).execute({"command": "sleep 3; echo late"})
    assert result["returncode"] == -1
    assert "longrun <command>" in result["exception_info"]
    assert "late" not in result["output"]


def test_longrun_outlives_normal_timeout_and_keeps_full_log():
    result = LongJobEnvironment(timeout=1).execute({"command": "longrun sh -c 'sleep 2; echo finished'"})
    assert result["returncode"] == 0
    assert "exited with code 0" in result["output"]
    assert "finished" in result["output"]
    assert log_path(result).read_text() == "finished\n"


def test_longrun_reports_failure_exit_code_and_stderr():
    result = LongJobEnvironment().execute({"command": "longrun sh -c 'echo to-stdout; echo to-stderr >&2; exit 3'"})
    assert result["returncode"] == 3
    assert "exited with code 3" in result["output"]
    assert "to-stdout" in result["output"]
    assert "to-stderr" in result["output"]


def test_longrun_is_killed_after_job_timeout():
    result = LongJobEnvironment(job_timeout=1).execute({"command": "longrun sh -c 'echo started; sleep 30'"})
    assert result["returncode"] < 0
    assert "killed after exceeding job_timeout (1s)" in result["output"]
    assert "started" in result["output"]


def test_longrun_returns_only_log_tail_for_big_output():
    result = LongJobEnvironment(log_tail_bytes=100).execute({"command": "longrun seq 1 5000"})
    assert f"showing only the last 100 of {log_path(result).stat().st_size} bytes" in result["output"]
    assert result["output"].endswith("4999\n5000\n")
    assert len(result["output"]) < 400
    assert log_path(result).read_text().splitlines()[0] == "1"
    assert len(log_path(result).read_text().splitlines()) == 5000


def test_longrun_uses_configured_cwd_and_env(tmp_path):
    env = LongJobEnvironment(cwd=str(tmp_path), env={"LONGJOB_VAR": "hello"})
    result = env.execute({"command": "longrun sh -c 'pwd; echo $LONGJOB_VAR'"})
    assert str(tmp_path.resolve()) in result["output"]
    assert "hello" in result["output"]


def test_each_longrun_gets_its_own_log():
    env = LongJobEnvironment()
    first, second = env.execute({"command": "longrun echo one"}), env.execute({"command": "longrun echo two"})
    assert (log_path(first).read_text(), log_path(second).read_text()) == ("one\n", "two\n")


def test_longjob_config_works_with_interactive_agent():
    config = recursive_merge(get_config_from_spec("mini.yaml"), get_config_from_spec("long_jobs.yaml"))
    model = DeterministicModel(
        outputs=[
            make_output("run it", [{"command": "longrun sh -c 'sleep 2; echo build ok'"}]),
            make_output("done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho analysis"}]),
        ]
    )
    env = get_environment(config["environment"] | {"timeout": 1}, default_type="local")
    agent = get_agent(model, env, config["agent"] | {"mode": "yolo", "confirm_exit": False}, default_type="interactive")
    assert agent.run("run the slow build") == {"exit_status": "Submitted", "submission": "analysis\n"}
    assert isinstance(env, LongJobEnvironment)
    assert "`longrun` prefix" in agent.messages[0]["content"]
    assert "at most 3600 seconds" in agent.messages[0]["content"]
    observation = agent.messages[3]["content"]
    assert "exited with code 0" in observation
    assert "build ok" in observation
