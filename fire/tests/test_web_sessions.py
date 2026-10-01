import time

from fire.web.app import Sessions


def test_login_survives_restart_and_expires(tmp_path):
    path = tmp_path / "state" / "sessions.json"
    s = Sessions(path, hours=8)
    token = s.new()
    assert s.valid(token) and not s.valid("other") and not s.valid(None)
    assert token not in path.read_text(encoding="utf-8")         # only hashes on disk
    assert Sessions(path).valid(token)                            # app restarted / deployed
    Sessions(path).drop(token)
    assert not Sessions(path).valid(token)                        # logout


def test_expired_session_is_rejected(tmp_path):
    s = Sessions(tmp_path / "sessions.json", hours=1 / 3600)      # 1 s
    token = s.new()
    time.sleep(1.1)
    assert not s.valid(token)


def test_broken_file_means_logged_out(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text("not json", encoding="utf-8")
    assert not Sessions(path).valid("x")
