import logging
import gc
import os

import numpy as np
from faster_whisper import WhisperModel

from translator import LANGUAGE_DISPLAY

log = logging.getLogger("LiveTranslate.ASR")

SAMPLE_RATE = 16000
DEFAULT_PAD_SECONDS = 0.5
PAD_SECONDS_ENV = "LIVETRANS_WHISPER_PAD_SECONDS"

LANGUAGE_NAMES = {**LANGUAGE_DISPLAY, "auto": "auto"}


class ASREngine:
    """Speech-to-text using faster-whisper."""

    def __init__(
        self,
        model_size="medium",
        device="cuda",
        device_index=0,
        compute_type="float16",
        language="auto",
        download_root=None,
        pad_seconds=None,
    ):
        self.language = language if language != "auto" else None
        if str(device).startswith("cuda"):
            # ctranslate2 does not ship cublas64_12.dll on Windows. Importing
            # torch first loads the matching CUDA 12 runtime from the nvidia-*
            # wheel packages into the process, so ctranslate2 can find it.
            try:
                import torch  # noqa: F401
            except Exception:
                log.warning("Failed to preload torch CUDA runtime", exc_info=True)
        model_kwargs = dict(
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            download_root=download_root,
        )
        # faster-whisper calls huggingface_hub.snapshot_download() even for an
        # already cached model. With an offline/broken proxy that HEAD request
        # fails before ctranslate2 can touch CUDA, so try the local cache first
        # and only fall back to a real download when the model is absent.
        try:
            self._model = WhisperModel(model_size, local_files_only=True, **model_kwargs)
            local_only = True
        except Exception as local_exc:
            log.warning(
                f"Whisper local cache unavailable ({local_exc}); downloading",
                exc_info=True,
            )
            self._model = WhisperModel(model_size, local_files_only=False, **model_kwargs)
            local_only = False
        self._set_input_padding(pad_seconds, log_change=False)
        log.info(
            f"Model loaded: {model_size} on {device} ({compute_type}), "
            f"local_only={local_only}"
        )
        self._log_input_padding()

    @staticmethod
    def _read_pad_seconds(value=None) -> float:
        if value is None:
            value = os.environ.get(PAD_SECONDS_ENV)
        if value is None:
            return DEFAULT_PAD_SECONDS
        try:
            return float(value)
        except (TypeError, ValueError):
            log.warning(
                f"Invalid Whisper pad seconds={value!r}; "
                f"using default {DEFAULT_PAD_SECONDS:g}s"
            )
            return DEFAULT_PAD_SECONDS

    def _set_input_padding(self, pad_seconds=None, log_change=True):
        self._pad_seconds = self._read_pad_seconds(pad_seconds)
        self._pad_quantum = int(round(SAMPLE_RATE * self._pad_seconds))
        if log_change:
            self._log_input_padding()

    def _log_input_padding(self):
        if self._pad_quantum > 0:
            log.info(
                "Whisper input padding enabled: "
                f"bucket={self._pad_seconds:g}s, quantum={self._pad_quantum} samples"
            )
        else:
            log.info("Whisper input padding disabled")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"ASR language: {old} -> {self.language}")

    def set_input_padding(self, pad_seconds):
        old_quantum = self._pad_quantum
        self._set_input_padding(pad_seconds, log_change=False)
        if self._pad_quantum != old_quantum:
            self._log_input_padding()

    def to_device(self, device: str):
        # ctranslate2 doesn't support device migration; must reload
        return False

    def unload(self):
        model = self._model
        self._model = None
        if model is None:
            return

        for attr in ("model", "feature_extractor", "hf_tokenizer"):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except Exception as e:
                    log.debug(f"Failed to delete WhisperModel.{attr}: {e}")
        del model
        gc.collect()

    def _prepare_audio_input(self, audio: np.ndarray) -> np.ndarray:
        if self._pad_quantum <= 0 or audio.size == 0:
            return audio

        original_samples = audio.shape[0]
        remainder = original_samples % self._pad_quantum
        if remainder == 0:
            return audio

        padded_samples = original_samples + self._pad_quantum - remainder
        padded = np.pad(audio, (0, padded_samples - original_samples), mode="constant")
        log.debug(f"Whisper input padded: {original_samples} -> {padded_samples} samples")
        return padded

    def transcribe(self, audio: np.ndarray, word_timestamps: bool = False) -> dict | None:
        """Transcribe audio segment.

        Args:
            audio: float32 numpy array, 16kHz mono
            word_timestamps: if True, include per-word timestamps in result

        Returns:
            dict with 'text', 'language', 'language_name' (and 'words' if word_timestamps) or None.
        """
        audio_input = self._prepare_audio_input(audio)
        segments, info = self._model.transcribe(
            audio_input,
            language=self.language,
            beam_size=5,
            vad_filter=False,
            word_timestamps=word_timestamps,
        )

        text_parts = []
        words = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            if word_timestamps and seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})

        full_text = " ".join(text_parts).strip()
        if not full_text:
            return None

        detected_lang = info.language
        result = {
            "text": full_text,
            "language": detected_lang,
            "language_name": LANGUAGE_NAMES.get(detected_lang, detected_lang),
        }
        if word_timestamps and words:
            result["words"] = words
        return result
