import io
import logging
import sys

import pytest

import main


def _record(msg="hello"):
    return logging.LogRecord("test", logging.INFO, __file__, 1, msg, None, None)


class _DeadStream:
    def write(self, _s):
        raise ValueError("I/O operation on closed file")

    def flush(self):
        pass


def test_no_console_at_all_keeps_handler_quiet(monkeypatch):
    """pythonw.exe gives us sys.stdout is None and sys.stderr is None."""
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    handler = main._SafeConsoleHandler()

    assert handler.stream is None
    handler.emit(_record())  # must not raise / must not print a Logging error


def test_console_dying_mid_run_disables_only_console(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _DeadStream())
    handler = main._SafeConsoleHandler()
    assert handler.stream is not None

    handler.emit(_record("first"))

    assert handler.stream is None
    handler.emit(_record("second"))  # already disabled, still silent


def test_working_console_still_prints(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)

    handler = main._SafeConsoleHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.emit(_record("visible"))

    assert buf.getvalue().strip() == "visible"


@pytest.mark.parametrize("attr", ["stdout", "stderr"])
def test_stderr_is_used_when_stdout_is_missing(monkeypatch, attr):
    """A detached stdout should still leave console output on stderr."""
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", err)

    handler = main._SafeConsoleHandler()

    assert handler.stream is err
