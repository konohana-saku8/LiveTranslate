"""Remote subtitle server for the Android client.

The PC captures a phone's playback audio streamed over a websocket, runs it
through the same Silero VAD + ASR + streaming translator pipeline as the local
capture path, and streams JSON subtitle events back.

Protocol (binary frames = raw PCM16LE, 16 kHz, mono):

    -> {"type": "ping"}
    <- {"type": "pong"}
    <- {"type": "ready", "sample_rate": 16000, "target_language": "zh"}
    <- {"type": "asr", "text": ..., "language": ..., "asr_ms": ..., "timestamp": ...}
    <- {"type": "translation_delta", "text": ...}
    <- {"type": "translation", "text": ..., "tl_ms": ...}
    <- {"type": "error", "message": ...}

Only one client may be connected at a time; further connections are closed
with code 4409 instead of kicking the active one.
"""

import json
import logging
import threading
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from vad_processor import VADProcessor

log = logging.getLogger("LiveTranslate.Remote")

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32 ms, Silero VAD's native window
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8766

CLOSE_UNAUTHORIZED = 4401
CLOSE_BUSY = 4409


class RemoteServer:
    """Websocket subtitle server bound to the running LiveTranslate pipeline.

    All heavy callbacks (ASR / translation) are supplied by the app so the
    remote path shares the loaded models instead of loading a second copy.
    """

    def __init__(
        self,
        asr_fn,
        translate_iter_fn,
        target_lang_fn,
        vad_settings_fn,
        overlay=None,
        sync_overlay=False,
        on_status=None,
        history_log_fn=None,
    ):
        self._asr_fn = asr_fn
        self._translate_iter_fn = translate_iter_fn
        self._target_lang_fn = target_lang_fn
        self._vad_settings_fn = vad_settings_fn
        self._overlay = overlay
        self._sync_overlay = sync_overlay
        self._on_status = on_status
        # Mirrors the local pipeline: every finished sentence pair is written,
        # failed translations included. Same-language lines are not stored.
        self._history_log_fn = history_log_fn

        self.host = DEFAULT_HOST
        self.port = DEFAULT_PORT
        self.token = ""

        self._lock = threading.Lock()
        self._server = None
        self._thread = None
        self._running = False
        self._clients = 0
        self._vad = None
        self._msg_id = 0

    # ── lifecycle ──

    def start(self, host=DEFAULT_HOST, port=DEFAULT_PORT, token="", sync_overlay=False):
        if self._running:
            return
        self.host = host or DEFAULT_HOST
        self.port = int(port)
        self.token = (token or "").strip()
        self._sync_overlay = bool(sync_overlay)

        vad_settings = {}
        try:
            vad_settings = self._vad_settings_fn() or {}
        except Exception:
            log.warning("Failed to read VAD settings for remote server", exc_info=True)
        self._vad = VADProcessor(
            sample_rate=SAMPLE_RATE,
            threshold=vad_settings.get("vad_threshold", 0.5),
            min_speech_duration=vad_settings.get("min_speech_duration", 1.0),
            max_speech_duration=vad_settings.get("max_speech_duration", 8.0),
            chunk_duration=CHUNK_SAMPLES / SAMPLE_RATE,
        )
        if vad_settings:
            self._vad.update_settings(vad_settings)

        # Binds immediately, so a port conflict surfaces to the caller.
        self._server = serve(
            self._handle,
            self.host,
            self.port,
            compression=None,  # PCM does not compress; skip the CPU cost
            max_size=None,
        )
        self._running = True
        self._thread = threading.Thread(
            target=self._serve, name="RemoteServer", daemon=True
        )
        self._thread.start()
        log.info(f"Remote server listening on {self.host}:{self.port}")
        self._report_status()

    def _serve(self):
        try:
            self._server.serve_forever()
        except Exception:
            if self._running:
                log.error("Remote server loop failed", exc_info=True)
        finally:
            self._running = False
            log.info("Remote server stopped")
            self._report_status()

    def stop(self):
        if not self._running:
            return
        self._running = False
        server = self._server
        self._server = None
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                log.warning("Remote server shutdown failed", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._vad = None
        log.info("Remote server stop requested")
        self._report_status()

    @property
    def running(self) -> bool:
        return self._running

    def update_vad_settings(self, settings: dict):
        with self._lock:
            if self._vad is not None:
                try:
                    self._vad.update_settings(settings)
                except Exception:
                    log.warning("Remote VAD update failed", exc_info=True)

    def status(self) -> dict:
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "clients": self._clients,
        }

    def _report_status(self):
        if self._on_status is None:
            return
        try:
            self._on_status(self.status())
        except Exception:
            log.debug("Remote status callback failed", exc_info=True)

    # ── connection handling ──

    def _handle(self, conn):
        try:
            remote = conn.remote_address
        except OSError:
            remote = "?"
        path = getattr(getattr(conn, "request", None), "path", "") or ""
        token = parse_qs(urlparse(path).query).get("token", [""])[0]
        if self.token and token != self.token:
            log.warning(f"Remote client rejected: bad token from {remote}")
            conn.close(CLOSE_UNAUTHORIZED, "unauthorized")
            return

        # A quick reconnect after a drop can race with the old handler's
        # cleanup; give it a moment to release the slot before rejecting.
        claimed = False
        deadline = time.monotonic() + 2.0
        while True:
            with self._lock:
                if self._clients == 0:
                    self._clients = 1
                    claimed = True
                    break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if not claimed:
            log.warning(f"Remote client rejected: already active ({remote})")
            conn.close(CLOSE_BUSY, "busy")
            return

        log.info(f"Remote client connected: {remote}")
        self._report_status()
        try:
            self._session(conn)
        except ConnectionClosed:
            log.info("Remote client closed the connection")
        except Exception:
            log.warning("Remote session ended with error", exc_info=True)
        finally:
            with self._lock:
                self._clients = 0
                if self._vad is not None:
                    try:
                        self._vad.force_flush()
                    except Exception:
                        pass
            log.info(f"Remote client disconnected: {remote}")
            self._report_status()

    def _session(self, conn):
        self._send(conn, {
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "target_language": self._safe_target_lang(),
        })

        leftover = np.zeros(0, dtype=np.float32)
        pending = b""
        while self._running:
            try:
                data = conn.recv(timeout=0.5)
            except TimeoutError:
                self._flush_on_timeout(conn)
                continue
            if data is None:
                break
            if isinstance(data, str):
                if not self._handle_text(conn, data):
                    break
                continue

            pending += data
            usable = len(pending) - (len(pending) % 2)
            if usable <= 0:
                continue
            samples = np.frombuffer(pending[:usable], dtype="<i2").astype(np.float32)
            pending = pending[usable:]
            leftover = np.concatenate([leftover, samples / 32768.0])

            while len(leftover) >= CHUNK_SAMPLES:
                chunk = leftover[:CHUNK_SAMPLES]
                leftover = leftover[CHUNK_SAMPLES:]
                self._feed(chunk, conn)
                if not self._running:
                    return

    def _handle_text(self, conn, data: str) -> bool:
        """Returns False when the client asked to end the session."""
        try:
            message = json.loads(data)
        except (TypeError, ValueError):
            return True
        if message.get("type") == "ping":
            self._send(conn, {"type": "pong"})
        elif message.get("type") == "stop":
            return False
        return True

    def _feed(self, chunk: np.ndarray, conn):
        vad = self._vad
        if vad is None:
            return
        try:
            segment = vad.process_chunk(chunk)
        except Exception:
            log.error("Remote VAD failed", exc_info=True)
            return
        if segment is not None:
            self._process_segment(conn, segment)

    def _flush_on_timeout(self, conn):
        """Treat a stalled stream like silence so the last sentence is emitted."""
        vad = self._vad
        if vad is None or not vad._is_speaking:
            return
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        for _ in range(vad._get_effective_silence_limit() + 1):
            try:
                segment = vad.process_chunk(silence)
            except Exception:
                return
            if segment is not None:
                self._process_segment(conn, segment)
                return

    # ── pipeline ──

    def _process_segment(self, conn, segment: np.ndarray):
        try:
            result, asr_ms = self._asr_fn(segment)
        except Exception as e:
            log.error("Remote ASR failed", exc_info=True)
            self._send_error(conn, f"ASR error: {e}")
            return
        if asr_ms == 0.0 or result is None:
            self._send_error(conn, "ASR 未就绪")
            return

        text = (result.get("text") or "").strip()
        if not text or not any(c.isalnum() for c in text):
            return
        seg_len = len(segment) / SAMPLE_RATE
        alnum = sum(1 for c in text if c.isalnum())
        if seg_len >= 2.0 and alnum <= 3:
            return

        lang = result.get("language", "auto")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._msg_id += 1
        msg_id = -self._msg_id  # keep remote ids out of the local id space

        log.info(f"Remote ASR [{lang}] ({asr_ms:.0f}ms): {text}")
        self._send(conn, {
            "type": "asr",
            "text": text,
            "language": lang,
            "asr_ms": round(asr_ms, 1),
            "timestamp": timestamp,
        })
        if self._sync_overlay and self._overlay is not None:
            try:
                self._overlay.add_message(msg_id, timestamp, text, lang, asr_ms)
            except Exception:
                log.debug("Overlay mirror (asr) failed", exc_info=True)

        target = self._safe_target_lang()
        if lang == target:
            self._send(conn, {"type": "translation", "text": "", "tl_ms": 0.0})
            if self._sync_overlay and self._overlay is not None:
                try:
                    self._overlay.update_translation(msg_id, "", 0)
                except Exception:
                    pass
            return

        tl_start = time.perf_counter()
        translated = ""
        failed = False
        try:
            for partial in self._translate_iter_fn(text, lang):
                translated = partial
                self._send(conn, {"type": "translation_delta", "text": partial})
                if self._sync_overlay and self._overlay is not None:
                    try:
                        self._overlay.update_streaming(msg_id, partial)
                    except Exception:
                        pass
            tl_ms = (time.perf_counter() - tl_start) * 1000
        except Exception as e:
            log.warning(f"Remote translate error: {e}")
            translated = f"[error: {e}]"
            failed = True
            tl_ms = (time.perf_counter() - tl_start) * 1000

        self._send(conn, {
            "type": "translation",
            "text": translated,
            "tl_ms": round(tl_ms, 1),
        })
        if self._sync_overlay and self._overlay is not None:
            try:
                self._overlay.update_translation(msg_id, translated, tl_ms)
            except Exception:
                log.debug("Overlay mirror (translation) failed", exc_info=True)
        if self._history_log_fn is not None:
            try:
                self._history_log_fn(text, translated, target, failed or not translated)
            except Exception:
                log.debug("Remote history write failed", exc_info=True)

    def _safe_target_lang(self) -> str:
        try:
            return self._target_lang_fn() or "zh"
        except Exception:
            return "zh"

    # ── io helpers ──

    def _send(self, conn, payload: dict):
        try:
            conn.send(json.dumps(payload, ensure_ascii=False))
        except Exception:
            # Client went away mid-sentence; the recv loop will notice.
            pass

    def _send_error(self, conn, message: str):
        self._send(conn, {"type": "error", "message": message})
