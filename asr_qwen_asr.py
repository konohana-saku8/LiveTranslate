import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("LiveTranslate.QwenASR")

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"

# Qwen3-ASR expects full language names (or its own codes). Map our short
# codes to the names used by the model card so forced-language works reliably.
_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "ro": "Romanian",
}


class QwenASREngine:
    """Speech-to-text using Qwen3-ASR-1.7B (Transformers native audio-LLM).

    Loaded through transformers AutoProcessor + AutoModelForMultimodalLM.
    The checkpoint may live outside the project's model cache, so an explicit
    path can be supplied; otherwise the HF cache / repo id is used.
    """

    def __init__(self, device="cuda", hub="hf", model_path: str | None = None):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        self._device = device
        self._language = "ja"
        self.model_path = self._resolve_model_path(model_path, hub)

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path, dtype=dtype
        ).to(device)
        self._model.eval()
        log.info(
            f"Qwen3-ASR loaded from {self.model_path} on {device} (dtype={dtype})"
        )

    def _resolve_model_path(self, model_path, hub) -> str:
        """Prefer an explicit local dir, then the cache, then the HF repo id."""
        if model_path:
            p = Path(model_path).expanduser()
            if p.is_dir() and (p / "config.json").is_file():
                return str(p.resolve())
            log.warning(f"Qwen3-ASR path not found or invalid: {model_path}")

        try:
            from model_manager import get_local_model_path

            cached = get_local_model_path("qwen-asr", hub=hub)
            if cached:
                return cached
        except Exception:
            pass
        return MODEL_ID

    def set_language(self, language: str):
        if language and language != "auto":
            self._language = language

    def to_device(self, device: str):
        try:
            import torch

            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            self._model.to(device, dtype=dtype)
            self._device = device
            log.info(f"Qwen3-ASR moved to {device}")
            return True
        except Exception as e:
            log.warning(f"Qwen3-ASR to_device failed: {e}")
            return False

    def unload(self):
        try:
            import torch

            self._model.to("cpu")
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._model = None

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict | None:
        """Transcribe audio segment (float32, 16kHz mono)."""
        import torch

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        lang = _LANGUAGE_NAMES.get(str(self._language).lower(), self._language)
        # "auto" or unknown -> let the model detect the language.
        language = None if str(self._language).lower() in ("auto", "none", "") else lang

        inputs = self._processor.apply_transcription_request(
            audio=audio, language=language
        ).to(self._model.device, self._model.dtype)
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=512)
        generated = output[:, inputs["input_ids"].shape[1]:]
        text = self._processor.decode(
            generated, return_format="transcription_only"
        )[0].strip()
        if not text:
            return None

        return {
            "text": text,
            "language": self._language,
            "language_name": self._language,
        }
