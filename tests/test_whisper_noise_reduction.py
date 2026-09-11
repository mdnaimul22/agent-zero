import asyncio
import io
import sys
import wave
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins._whisper_stt.helpers import runtime


def test_config_normalization():
    cfg_default = runtime.normalize_config({})
    assert "noise_reduction" in cfg_default
    assert cfg_default["noise_reduction"] is False

    cfg_enabled = runtime.normalize_config({"noise_reduction": True})
    assert cfg_enabled["noise_reduction"] is True

    cfg_disabled = runtime.normalize_config({"noise_reduction": False})
    assert cfg_disabled["noise_reduction"] is False


def test_apply_noise_reduction_synthetic_audio():
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * np.random.normal(size=t.shape)
    noisy_audio = (signal + noise).astype(np.float32)

    denoised = runtime._apply_noise_reduction(noisy_audio, sample_rate)
    assert isinstance(denoised, np.ndarray)
    assert len(denoised) == len(noisy_audio)
    assert not np.isnan(denoised).any()
    assert not np.isinf(denoised).any()


def test_apply_noise_reduction_fallback():
    sample_rate = 16000
    raw_array = np.zeros(1000, dtype=np.float32)
    result = runtime._apply_noise_reduction(raw_array, sample_rate)
    assert isinstance(result, np.ndarray)
    assert len(result) == 1000


def test_transcribe_noise_reduction_flag():
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767.0).astype(np.int16)

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())

    import base64
    audio_b64 = base64.b64encode(wav_io.getvalue()).decode("utf-8")

    async def _run():
        res_no_nr = await runtime.transcribe(
            audio_b64,
            config={"model_size": "tiny", "noise_reduction": False, "language": "en"}
        )
        assert isinstance(res_no_nr, dict)

        res_with_nr = await runtime.transcribe(
            audio_b64,
            config={"model_size": "tiny", "noise_reduction": True, "language": "en"}
        )
        assert isinstance(res_with_nr, dict)

    asyncio.run(_run())


if __name__ == "__main__":
    print("Running test_config_normalization...", flush=True)
    test_config_normalization()
    print("✓ test_config_normalization passed!", flush=True)

    print("Running test_apply_noise_reduction_synthetic_audio...", flush=True)
    test_apply_noise_reduction_synthetic_audio()
    print("✓ test_apply_noise_reduction_synthetic_audio passed!", flush=True)

    print("Running test_apply_noise_reduction_fallback...", flush=True)
    test_apply_noise_reduction_fallback()
    print("✓ test_apply_noise_reduction_fallback passed!", flush=True)

    print("Running test_transcribe_noise_reduction_flag...", flush=True)
    test_transcribe_noise_reduction_flag()
    print("✓ test_transcribe_noise_reduction_flag passed!", flush=True)

    print("\nALL NOISE REDUCTION TESTS PASSED!", flush=True)
