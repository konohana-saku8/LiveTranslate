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


def test_tee_mirrors_to_both_sides():
    terminal, captured = io.StringIO(), io.StringIO()
    tee = main._TeeStream(terminal, captured)

    tee.write("hello\n")
    tee.flush()

    assert terminal.getvalue() == "hello\n"
    assert captured.getvalue() == "hello\n"


def test_tee_survives_a_dead_terminal():
    class Dead:
        def write(self, _s):
            raise ValueError("I/O operation on closed file")

        def flush(self):
            pass

    captured = io.StringIO()
    tee = main._TeeStream(Dead(), captured)

    tee.write("still here\n")
    tee.write("and again\n")

    assert tee._primary is None
    assert captured.getvalue() == "still here\nand again\n"


def test_tee_without_any_terminal(capsys=None):
    """Under pythonw both sides would be None; the file side must still work."""
    captured = io.StringIO()
    tee = main._TeeStream(None, captured)

    tee.write("captured only\n")

    assert captured.getvalue() == "captured only\n"


def test_tee_flattens_progress_carriage_returns():
    captured = io.StringIO()
    tee = main._TeeStream(None, captured)

    tee.write("Fetching 7 files:  14%")
    tee.write("\rFetching 7 files:  29%\n")

    assert "\r" not in captured.getvalue()
    assert "14%" in captured.getvalue() and "29%" in captured.getvalue()


def test_tee_reports_non_tty_so_progress_bars_stay_readable():
    assert main._TeeStream(None, io.StringIO()).isatty() is False
    assert main._TeeStream(io.StringIO(), None).isatty() is False
