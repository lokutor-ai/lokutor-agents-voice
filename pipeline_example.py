"""Full VoicePipeline (STT -> workflow -> TTS) on Lokutor. The workflow here just echoes, so no LLM key is
needed; swap in SingleAgentVoiceWorkflow(agent) for a real agent."""
import asyncio
import numpy as np
from agents.voice import (AudioInput, VoicePipeline, VoicePipelineConfig, VoiceWorkflowBase,
                          TTSModelSettings)
from lokutor_agents_voice import LokutorVoiceModelProvider, LokutorTTSModel, SAMPLE_RATE

class Echo(VoiceWorkflowBase):
    async def run(self, transcription: str):
        yield f"You said: {transcription}"

async def main():
    provider = LokutorVoiceModelProvider(voice="F1", language="en")
    pcm = b"".join([c async for c in LokutorTTSModel(language="en").run("What time is it in Madrid?", TTSModelSettings())])
    pipe = VoicePipeline(workflow=Echo(), stt_model=provider.get_stt_model(None),
                         tts_model=provider.get_tts_model(None),
                         config=VoicePipelineConfig(tracing_disabled=True))
    result = await pipe.run(AudioInput(buffer=np.frombuffer(pcm, np.int16), frame_rate=SAMPLE_RATE))
    n = 0
    async for ev in result.stream():
        if ev.type == "voice_stream_event_audio" and ev.data is not None:
            n += len(ev.data)
    print(f"reply audio: {n/SAMPLE_RATE:.2f}s")

asyncio.run(main())
