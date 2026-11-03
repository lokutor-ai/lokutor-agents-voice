# lokutor-agents-voice

[Lokutor](https://lokutor.com) text-to-speech and speech-to-text for the
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) voice pipeline
(`VoicePipeline`).

Pointing the SDK's built-in OpenAI voice models at Lokutor with a base URL is not enough:

- the SDK's TTS model asks for raw 24 kHz PCM, Lokutor's OpenAI-compatible
  `/v1/audio/speech` always answers a 44.1 kHz WAV;
- the SDK's `transcribe()` uses the Realtime WebSocket API, which Lokutor does not offer.

This package provides `LokutorTTSModel`, `LokutorSTTModel` and `LokutorVoiceModelProvider`, which call
`POST /v1/audio/speech` and `POST /v1/audio/transcriptions` and convert the audio.

```bash
pip install git+https://github.com/lokutor-ai/lokutor-agents-voice
export LOKUTOR_API_KEY=...        # https://app.lokutor.com
python example.py                 # TTS -> STT round trip
python pipeline_example.py        # full VoicePipeline with an echo workflow
```

```python
from agents.voice import VoicePipeline, SingleAgentVoiceWorkflow
from lokutor_agents_voice import LokutorVoiceModelProvider

provider = LokutorVoiceModelProvider(voice="F1", language="en")   # voices F1-F5, M1-M5
pipeline = VoicePipeline(
    workflow=SingleAgentVoiceWorkflow(agent),
    stt_model=provider.get_stt_model(None),
    tts_model=provider.get_tts_model(None),
)
```

Limits: push-to-talk input only (`AudioInput`); `StreamedAudioInput` raises `NotImplementedError`
(use Lokutor's `/ws/stt` directly). TTS is one request per sentence chunk, not streamed audio.
Resampling is linear interpolation.

Verified against the live API with openai-agents 0.24.0: TTS -> STT round trip and an echo
`VoicePipeline` run. Not run: `SingleAgentVoiceWorkflow` with an LLM, `StreamedAudioInput`.

Docs: https://docs.lokutor.com/integrations/openai-compatible
