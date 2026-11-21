import sqlite3

from history_db import (
    FAILED_PLACEHOLDER,
    TranslationHistoryDB,
    debug_db_path,
    to_history_language,
)


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


def test_failed_translation_is_recorded_with_source_and_filterable_status(tmp_path):
    """A dead translation API must not lose the ASR text.

    Failed rows keep the Japanese source, carry a placeholder live-archive
    filters out, and are marked so they can be selected for manual correction.
    """
    db = TranslationHistoryDB(str(tmp_path / "history.db"), api_name="LiveTranslate:test")

    db.log("こんにちは", "[error: timeout]", "zh")          # UI error marker -> failed
    db.log("さようなら", "", "zh", failed=True)              # explicit failure
    db.log("おはよう", "早上好", "zh")                        # success
    db.log("", "你好", "zh")                                  # no source -> skipped
    db.close()

    rows = sqlite3.connect(str(tmp_path / "history.db")).execute(
        "SELECT SourceText, TranslatedText, Status FROM TranslationHistory ORDER BY Id"
    ).fetchall()
    assert rows == [
        ("こんにちは", FAILED_PLACEHOLDER, "failed"),
        ("さようなら", FAILED_PLACEHOLDER, "failed"),
        ("おはよう", "早上好", "ok"),
    ]
    # live-archive::clean_translation() blanks any row containing this text.
    assert "Translation Failed" in FAILED_PLACEHOLDER

    failed = sqlite3.connect(str(tmp_path / "history.db")).execute(
        "SELECT SourceText FROM TranslationHistory WHERE Status='failed'"
    ).fetchall()
    assert failed == [("こんにちは",), ("さようなら",)]
