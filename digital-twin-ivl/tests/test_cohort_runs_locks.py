"""Lock files of concurrent workers (dtivl.drivers.common.claim_lock)."""
import os
import subprocess
import sys
import time

from dtivl.drivers import common as C


def test_claim_release_and_held(tmp_path):
    lock = tmp_path / "a.lock"
    assert C.claim_lock(lock)
    assert lock.read_text() == str(os.getpid())          # pid only (readable by older workers)
    assert not C.claim_lock(lock)                         # held by this live Python process
    lock.unlink()


def test_dead_and_reused_pids_are_stale(tmp_path):
    lock = tmp_path / "b.lock"
    lock.write_text("999999999")                          # dead pid
    assert C.claim_lock(lock)
    # a live Python process started after the lock was written: the pid was reused (machine restart)
    lock.write_text("0")
    old = time.time() - 3600
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.write_text(str(p.pid))
        os.utime(lock, (old, old))
        assert C.claim_lock(lock)
        lock.write_text(str(p.pid))                       # written after the process started: held
        time.sleep(1.2)
        assert not C.claim_lock(lock)
    finally:
        p.kill()


def test_non_python_holder_is_stale(tmp_path):
    lock = tmp_path / "c.lock"
    p = subprocess.Popen(["sleep", "30"])
    try:
        time.sleep(0.2)
        lock.write_text(str(p.pid))
        assert C.claim_lock(lock)
    finally:
        p.kill()


def test_empty_lock_grace(tmp_path):
    lock = tmp_path / "d.lock"
    lock.write_text("")
    assert not C.claim_lock(lock, grace_s=60.0)
    assert C.claim_lock(lock, grace_s=0.0)
