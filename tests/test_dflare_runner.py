import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "experiments" / "run_dflare_gpu_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_dflare_gpu_experiment", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ENGINE = MODULE.ENGINE
_git_state = MODULE._git_state


def test_external_runner_is_kept_inside_integration_directory() -> None:
    assert ENGINE.name == "dflare_smoke.py"
    assert ENGINE.parent.name == "angelslim"
    assert ENGINE.is_file()


def test_git_state_records_immutable_revision() -> None:
    state = _git_state(PROJECT)
    assert len(state["commit"]) == 40
    assert isinstance(state["dirty"], bool)
