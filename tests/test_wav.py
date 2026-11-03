import io, wave
import numpy as np
from lokutor_agents_voice import wav_to_pcm16

def test_resamples_44k_to_24k():
    pcm = (np.sin(np.linspace(0, 100, 44100)) * 10000).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(pcm.tobytes())
    out = np.frombuffer(wav_to_pcm16(buf.getvalue()), dtype="<i2")
    assert len(out) == 24000
