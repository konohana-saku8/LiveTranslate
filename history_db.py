"""Translation history, appended to the shared SQLite database.

The C# LiveCaptions-Translator owns `translation_history.db` and its single
`TranslationHistory` table. This writer targets the same file and the same
schema so both programs contribute to one history: a row written here shows up
in that app's history page, and its rows are not disturbed.

The table is created when missing, so the first run of either program is
enough. Writes are best effort -- an unwritable database logs once and then
stays quiet. Losing a history row must never take down a translation.
"""

import logging
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("LiveTranslate.HistoryDB")

# Written into TranslatedText for a failed translation. The substring
# "Translation Failed" is what live-archive::clean_translation() looks for to
# blank out a row, so failed rows stay in the DB for manual correction without
# leaking into published transcripts.
FAILED_PLACEHOLDER = "[Translation Failed]"

# Identical to the statement in the C# HistoryLogger, so whichever program opens
# the file first produces the table the other one expects.
SCHEMA = """
CREATE TABLE IF NOT EXISTS TranslationHistory (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Timestamp TEXT,
    SourceText TEXT,
    TranslatedText TEXT,
    TargetLanguage TEXT,
    ApiUsed TEXT
);
"""

# Columns added after the C# schema was set. They describe *when the sentence was
# spoken* rather than when its translation came back, which is what a subtitle
# synced to a recording needs: `Timestamp` is written after ASR + translation, so
# it lags the voice by about a second (measured: VAD silence 0.4s + ASR 0.4s +
# translate 0.3s). ALTER TABLE ... ADD COLUMN is append-only, so the C# history
# page keeps reading the same table and simply sees NULLs in these columns.
#
#   SpeechStart  unix time of the first syllable (float, sub-second matters)
#   Duration     length of the speech segment in seconds
#   LagMs        spoken-end -> row written, for spotting queue hiccups
EXTRA_COLUMNS = (
    ("SpeechStart", "REAL"),
    ("Duration", "REAL"),
    ("LagMs", "INTEGER"),
    # 'ok' | 'failed'. Failed rows keep the Japanese source so it can be
    # corrected by hand later; the placeholder in TranslatedText is filtered
    # out by live-archive's export, so it never reaches a published transcript.
    ("Status", "TEXT"),
)


# The two programs name languages differently: this app speaks ISO 639-1 ("zh",
# "en") because that is what the LLM prompt and the UI menus use, while the C#
# app stores a BCP-47 tag ("zh-CN", "en-US") in the same column -- 4500+ of its
# rows already look like that. Translating at this one boundary keeps the column
# consistent without touching prompts, menus or saved user settings.
LANGUAGE_TAGS = {
    "zh": "zh-CN",
    "en": "en-US",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "fr": "fr-FR",
    "de": "de-DE",
    "es": "es-ES",
    "ru": "ru-RU",
    "pt": "pt-BR",
    "it": "it-IT",
    "tr": "tr-TR",
    "ar": "ar-SA",
    "th": "th-TH",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "vi": "vi-VN",
    "id": "id-ID",
    "ms": "ms-MY",
    "hi": "hi-IN",
    "uk": "uk-UA",
    "cs": "cs-CZ",
    "ro": "ro-RO",
    "el": "el-GR",
    "hu": "hu-HU",
    "sv": "sv-SE",
    "da": "da-DK",
    "fi": "fi-FI",
    "no": "nb-NO",
    "he": "he-IL",
}


def to_history_language(code: str) -> str:
    """Return the tag form the C# history stores; unknown codes pass through."""
    code = (code or "").strip()
    if not code:
        return ""
    if "-" in code:  # already a tag, e.g. someone set "zh-TW" by hand
        return code
    return LANGUAGE_TAGS.get(code.lower(), code)


def debug_db_path(production_path: str, override: str = "") -> str:
    """Path used while Debug is on. Never resolves to the production database.

    By default the debug file sits next to the real one with `.debug` inserted
    before the extension (`translation_history.db` -> `translation_history.debug.db`),
    so a test run cannot add rows to the archive the export tools read. An explicit
    override is honoured unless it points at the production file itself.
    """
    production = str(production_path or "").strip()
    custom = str(override or "").strip()
    if custom:
        try:
            same_as_production = bool(production) and (
                Path(custom).expanduser().resolve()
                == Path(production).expanduser().resolve()
            )
        except OSError:  # a bad path should not stop the writer from starting
            same_as_production = False
        if not same_as_production:
            return custom
    if not production:
        # No production DB configured means history is off; Debug must not
        # silently create a database somewhere else.
        return ""
    p = Path(production).expanduser()
    stem = p.stem if p.suffix else p.name
    if stem.endswith(".debug") or p.suffix == ".debug":
        return str(p)
    suffix = p.suffix or ".db"
    return str(p.with_name(f"{stem}.debug{suffix}"))


class TranslationHistoryDB:
    """Appends one row per finished sentence pair to the shared history."""

    def __init__(self, db_path, api_name="LiveTranslate"):
        self._path = Path(db_path).expanduser()
        self._api_name = api_name
        self._lock = threading.Lock()
        self._conn = None
        self._warned = False
        # Set by retire() when the app switches databases. A translation already
        # in flight may still hold a reference to this writer; without a permanent
        # stop flag it would reopen the closed connection and append to the *old*
        # database (i.e. pollute the production DB after Debug was enabled).
        self._retired = False

    @property
    def path(self) -> str:
        return str(self._path)

    def _connect(self):
        if self._conn is not None:
            return self._conn
        if self._retired:
            raise RuntimeError("history writer has been retired")

        # The directory may not exist yet when the C# app has never been built in
        # this location; creating it keeps the first write from failing.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), timeout=5.0, check_same_thread=False)
        # WAL lets the C# history page read while this process appends, and keeps
        # the two writers from serialising on a single lock. NORMAL is what the C#
        # side uses as well.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(SCHEMA)
        conn.commit()
        self._ensure_columns(conn)
        self._conn = conn
        return conn

    def _ensure_columns(self, conn):
        """Add the timing columns to a database created before they existed."""
        have = {row[1] for row in conn.execute("PRAGMA table_info(TranslationHistory)")}
        added = []
        for name, decl in EXTRA_COLUMNS:
            if name not in have:
                try:
                    conn.execute("ALTER TABLE TranslationHistory ADD COLUMN %s %s" % (name, decl))
                    added.append(name)
                except sqlite3.Error as exc:  # locked or read-only: keep working
                    log.warning("History DB: could not add column %s (%s)", name, exc)
        if added:
            conn.commit()
            log.info("History DB: added timing columns %s", ", ".join(added))

    def log(self, source: str, translated: str, target_language: str,
            speech_start: float = None, duration: float = None,
            failed: bool = False) -> None:
        """Append one sentence pair. Never raises.

        `speech_start` is when the sentence was actually spoken, `duration` how
        long it took. Both are optional so every existing caller keeps working;
        when they are missing the row is marked by NULL and readers fall back to
        `Timestamp` (which lags by roughly a second).

        A failed translation is recorded too (with the original source text and
        a placeholder), so a dead translation API cannot lose the Japanese the
        ASR produced. Set `failed=True` from the caller, or pass a bracketed
        placeholder as `translated` -- both mark the row Status='failed'. The
        placeholder contains "Translation Failed", which live-archive's export
        filters out, so these rows never become published subtitles.
        """
        source = (source or "").strip()
        translated = (translated or "").strip()
        if not source:
            return
        # Bracketed placeholders are the UI's failure markers, not translations.
        # Keep the row (the source is valuable) but flag it as failed.
        if translated.startswith("[") and translated.endswith("]"):
            failed = True
        if not translated:
            failed = True
        if failed:
            translated = FAILED_PLACEHOLDER

        now = time.time()
        lag_ms = None
        if speech_start and duration:
            lag_ms = int(max(0.0, now - (speech_start + duration)) * 1000)

        with self._lock:
            if self._retired:
                return
            try:
                self._connect().execute(
                    "INSERT INTO TranslationHistory "
                    "(Timestamp, SourceText, TranslatedText, TargetLanguage, ApiUsed, "
                    " SpeechStart, Duration, LagMs, Status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(int(now)),
                        source,
                        translated,
                        to_history_language(target_language),
                        self._api_name,
                        float(speech_start) if speech_start else None,
                        float(duration) if duration else None,
                        lag_ms,
                        "failed" if failed else "ok",
                    ),
                )
                self._connect().commit()
            except Exception as exc:  # a locked file must not stop the pipeline
                if not self._warned:
                    log.error(f"History DB write failed ({self._path}): {exc}")
                    self._warned = True

    def retire(self) -> None:
        """Close for good: used when a Debug toggle swaps the target database.

        Unlike close(), a retired writer will never reconnect, so a translation
        that was already routed to it cannot write to the previous file.
        """
        with self._lock:
            self._retired = True
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
