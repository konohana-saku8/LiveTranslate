import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("LiveTranslate.ReazonSpeech")

MODEL_ID = "reazon-research/reazonspeech-k2-v2"

# File stems shipped by the k2-v2 repo. fp32 and int8 are separate checkpoints;
# the precision option picks one set, it does not quantise on the fly.
_STEMS = {
    "encoder": "encoder-epoch-99-avg-1",
    "decoder": "decoder-epoch-99-avg-1",
    "joiner": "joiner-epoch-99-avg-1",
}


def normalize_reazonspeech_precision(value: str | None) -> str:
    v = str(value or "").strip().lower()
    return "int8" if v in ("int8", "quantized", "quant") else "fp32"


class ReazonSpeechEngine:
    """Japanese ASR via ReazonSpeech k2-v2 (sherpa-onnx transducer, CPU).

    Runs entirely on CPU: the model is only 159M params and int8 reaches ~36x
    realtime on one thread, so using the GPU would only steal VRAM from the
    translation model without any speed benefit. `device` is accepted for
    interface compatibility but ignored.
    """

    def __init__(self, precision: str = "int8", hub: str = "hf", device: str = "cpu"):
        import sherpa_onnx

        self.precision = normalize_reazonspeech_precision(precision)
        self._language = "ja"
        self.model_path = self._resolve_model_path(hub)

        suffix = "" if self.precision == "fp32" else ".int8"
        files = {k: str(Path(self.model_path) / f"{v}{suffix}.onnx")
                 for k, v in _STEMS.items()}
        files["tokens"] = str(Path(self.model_path) / "tokens.txt")
        missing = [f for f in files.values() if not Path(f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"ReazonSpeech {self.precision} model incomplete: {missing}"
            )

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=files["encoder"],
            decoder=files["decoder"],
            joiner=files["joiner"],
            tokens=files["tokens"],
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )
        log.info(
            f"ReazonSpeech loaded: {self.precision} from {self.model_path} (CPU)"
        )

    def _resolve_model_path(self, hub: str) -> str:
        try:
            from model_manager import get_local_model_path

            cached = get_local_model_path("reazonspeech", hub=hub)
            if cached:
                return cached
        except Exception:
            pass
        return str(Path(__file__).parent / "models" / "huggingface" / "hub"
                   / "models--reazon-research--reazonspeech-k2-v2" / "snapshots" / "master")

    def set_language(self, language: str):
        # k2-v2 is Japanese-only; keep the value for the result payload.
        if language and language != "auto":
            self._language = language

    def to_device(self, device: str):
        # CPU-only by design; nothing to move.
        return False

    def unload(self):
        self._recognizer = None

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict | None:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self._recognizer.decode_stream(stream)
        text = (stream.result.text or "").strip()
        if not text:
            return None
        return {
            "text": text,
            "language": self._language,
            "language_name": "Japanese",
        }
