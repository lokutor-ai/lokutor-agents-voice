"""TTS -> STT round trip through Lokutor with the Agents SDK model interfaces
(no OpenAI key needed). Set LOKUTOR_API_KEY."""
import asyncio
import numpy as np
from agents.voice import AudioInput, STTModelSettings, TTSModelSettings
from lokutor_agents_voice import LokutorSTTModel, LokutorTTSModel, SAMPLE_RATE

async def main():
    tts, stt = LokutorTTSModel(voice="F1", language="en"), LokutorSTTModel()
    pcm = b"".join([c async for c in tts.run("Hello from Lokutor, this is a test.", TTSModelSettings())])
    audio = np.frombuffer(pcm, dtype=np.int16)
    print(f"tts: {len(audio)/SAMPLE_RATE:.2f}s of 24 kHz PCM16")
    text = await stt.transcribe(AudioInput(buffer=audio, frame_rate=SAMPLE_RATE),
                                STTModelSettings(language="en"), False, False)
    print("stt:", text)

asyncio.run(main())
