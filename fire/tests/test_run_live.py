import pytest

from fire.run_live import qwen_enabled


@pytest.mark.parametrize("env, flag, expected", [
    (None, False, True),          # default: Qwen on
    ("on", False, True),
    ("off", False, False),        # FIRE_QWEN=off in fire/.env (droplet: qwen.sh off)
    (" OFF ", False, False),
    ("on", True, False),          # --no-qwen always wins
])
def test_qwen_switch(monkeypatch, env, flag, expected):
    if env is None:
        monkeypatch.delenv("FIRE_QWEN", raising=False)
    else:
        monkeypatch.setenv("FIRE_QWEN", env)
    assert qwen_enabled(flag) is expected
