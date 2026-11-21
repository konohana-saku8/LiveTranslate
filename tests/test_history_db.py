import sqlite3

from history_db import TranslationHistoryDB, debug_db_path, to_history_language


def test_short_codes_become_the_tags_the_csharp_history_uses():
    assert to_history_language("zh") == "zh-CN"
    assert to_history_language("ja") == "ja-JP"
    assert to_history_language("ZH") == "zh-CN"


def test_existing_tags_and_unknown_codes_pass_through():
    assert to_history_language("zh-TW") == "zh-TW"
    assert to_history_language("xx") == "xx"
    assert to_history_language("") == ""


def test_log_writes_one_row_with_the_tag_form(tmp_path):
    db = TranslationHistoryDB(str(tmp_path / "history.db"), api_name="LiveTranslate:test")

    db.log("こんにちは", "你好", "zh")
    db.close()

    rows = sqlite3.connect(str(tmp_path / "history.db")).execute(
        "SELECT TargetLanguage, ApiUsed FROM TranslationHistory"
    ).fetchall()
    assert rows == [("zh-CN", "LiveTranslate:test")]


def test_debug_path_is_a_sibling_and_never_the_production_file(tmp_path):
    prod = str(tmp_path / "translation_history.db")
    debug = debug_db_path(prod)
    assert debug == str(tmp_path / "translation_history.debug.db")
    assert debug != prod

    # An explicit override is honoured, unless it points back at the real DB.
    custom = str(tmp_path / "my_debug.db")
    assert debug_db_path(prod, custom) == custom
    assert debug_db_path(prod, prod) == debug
    assert debug_db_path("") == ""


def test_retired_writer_cannot_write_to_the_old_database(tmp_path):
    prod = str(tmp_path / "translation_history.db")
    debug = debug_db_path(prod)

    old = TranslationHistoryDB(prod)
    old.log("本番", "production", "zh")
    old.retire()
    # Simulates a translation that was already routed to the old writer when
    # Debug was switched on: it must not reopen and append to the archive.
    old.log("遗漏", "must not appear", "zh")

    new = TranslationHistoryDB(debug)
    new.log("テスト", "debug only", "zh")
    new.close()

    def rows(path):
        return sqlite3.connect(str(path)).execute(
            "SELECT SourceText FROM TranslationHistory ORDER BY Id"
        ).fetchall()

    assert rows(prod) == [("本番",)]
    assert rows(debug) == [("テスト",)]


def test_error_placeholder_is_not_history(tmp_path):
    db = TranslationHistoryDB(str(tmp_path / "history.db"), api_name="LiveTranslate:test")

    db.log("こんにちは", "[error: timeout]", "zh")
    db.log("", "你好", "zh")
    db.close()

    conn = sqlite3.connect(str(tmp_path / "history.db"))
    # The table is created lazily on the first real write, so a run that only
    # produced placeholders may leave no table at all.
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='TranslationHistory'"
    ).fetchone()
    count = (
        conn.execute("SELECT COUNT(*) FROM TranslationHistory").fetchone()[0]
        if table
        else 0
    )
    assert count == 0
