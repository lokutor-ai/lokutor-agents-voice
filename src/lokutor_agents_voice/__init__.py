"""Lokutor speech models for the OpenAI Agents SDK voice pipeline.

The SDK's built-in OpenAI models cannot simply be pointed at Lokutor with a base URL:
  * the TTS model asks for raw PCM at 24 kHz, Lokutor always answers WAV at 44.1 kHz;
  * the STT model's ``transcribe`` uses the Realtime WebSocket API, which Lokutor's
    OpenAI-compatible surface does not offer.
These two classes talk to POST /v1/audio/speech and POST /v1/audio/transcriptions instead.
"""
from __future__ import annotations

import io
import os
import wave
from collections.abc import AsyncIterator

import httpx
import numpy as np
from agents.voice import (
    AudioInput,
    StreamedAudioInput,
    STTModel,
    STTModelSettings,
    StreamedTranscriptionSession,
    TTSModel,
    TTSModelSettings,
    VoiceModelProvider,
)

BASE_URL = "https://api.lokutor.com/v1"
SAMPLE_RATE = 24000  # what VoicePipeline's output consumers expect by default


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = int(round(x.size * dst / src))
    xs = np.linspace(0, x.size - 1, n)
    return np.interp(xs, np.arange(x.size), x.astype(np.float32)).astype(np.int16)


def wav_to_pcm16(data: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Decode a mono 16-bit WAV and return raw PCM16 at ``rate`` Hz."""
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit WAV from Lokutor")
        ch = w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
        return _resample(pcm, w.getframerate(), rate).tobytes()


class LokutorTTSModel(TTSModel):
    def __init__(self, *, api_key: str | None = None, base_url: str = BASE_URL,
                 voice: str = "F1", language: str | None = None, model: str = "versa",
                 client: httpx.AsyncClient | None = None):
        self._key = api_key or os.environ["LOKUTOR_API_KEY"]
        self._url = base_url.rstrip("/") + "/audio/speech"
        self._voice, self._language, self._model = voice, language, model
        self._client = client or httpx.AsyncClient(timeout=60)

    @property
    def model_name(self) -> str:
        return self._model

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        body: dict = {"model": self._model, "input": text,
                      # a Lokutor voice (F1-F5, M1-M5) or an OpenAI name (alloy, ...) both work
                      "voice": settings.voice if isinstance(settings.voice, str) else self._voice}
        if settings.speed is not None:
            body["speed"] = settings.speed
        if self._language:
            body["language"] = self._language
        r = await self._client.post(self._url, json=body,
                                    headers={"Authorization": f"Bearer {self._key}"})
        r.raise_for_status()
        pcm = wav_to_pcm16(r.content)
        for i in range(0, len(pcm), 4096):
            yield pcm[i:i + 4096]


class LokutorSTTModel(STTModel):
    def __init__(self, *, api_key: str | None = None, base_url: str = BASE_URL,
                 model: str = "lokutor-stt", client: httpx.AsyncClient | None = None):
        self._key = api_key or os.environ["LOKUTOR_API_KEY"]
        self._url = base_url.rstrip("/") + "/audio/transcriptions"
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=60)

    @property
    def model_name(self) -> str:
        return self._model

    async def transcribe(self, input: AudioInput, settings: STTModelSettings,
                         trace_include_sensitive_data: bool,
                         trace_include_sensitive_audio_data: bool) -> str:
        # AudioInput.to_audio_file() yields a 16-bit PCM WAV, the only format Lokutor STT takes.
        files = {"file": input.to_audio_file()}
        data = {"model": self._model}
        if settings.language:
            data["language"] = settings.language
        r = await self._client.post(self._url, files=files, data=data,
                                    headers={"Authorization": f"Bearer {self._key}"})
        r.raise_for_status()
        return r.json()["text"]

    async def create_session(self, input: StreamedAudioInput, settings: STTModelSettings,
                             trace_include_sensitive_data: bool,
                             trace_include_sensitive_audio_data: bool) -> StreamedTranscriptionSession:
        raise NotImplementedError(
            "Streaming input (StreamedAudioInput) is not implemented in this adapter; "
            "use AudioInput (push-to-talk) or Lokutor's /ws/stt directly.")


class LokutorVoiceModelProvider(VoiceModelProvider):
    """Drop-in for ``VoicePipelineConfig(model_provider=...)``."""

    def __init__(self, *, api_key: str | None = None, base_url: str = BASE_URL,
                 voice: str = "F1", language: str | None = None):
        self._kw = dict(api_key=api_key, base_url=base_url)
        self._voice, self._language = voice, language

    def get_stt_model(self, model_name: str | None) -> STTModel:
        return LokutorSTTModel(**self._kw)

    def get_tts_model(self, model_name: str | None) -> TTSModel:
        return LokutorTTSModel(**self._kw, voice=self._voice, language=self._language)
