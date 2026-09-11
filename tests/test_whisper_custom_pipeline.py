import asyncio
import base64
import io
import json
import sys
import threading
import wave
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plugins._whisper_stt.helpers.runtime as runtime
from plugins._whisper_stt.api.transcribe import Transcribe
from helpers.api import Request, Response


def create_dummy_wav_base64(duration: float = 1.0, sample_rate: int = 16000) -> str:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return base64.b64encode(wav_io.getvalue()).decode("utf-8")


async def run_tests():
    print("=== Starting Comprehensive Whisper Pipeline & HTTP Tests ===")
    audio_b64 = create_dummy_wav_base64()
    handler = Transcribe(app=None, thread_lock=threading.Lock())

    print("\n[Test 1] Testing runtime.transcribe with shhossain/whisper-base-bn...")
    config_custom = {
        "model_size": "shhossain/whisper-base-bn",
        "custom_model": "",
        "language": "bn",
    }
    result_custom = await runtime.transcribe(audio_b64, config=config_custom)
    print("Result 1 (Custom HF):", result_custom)
    assert isinstance(result_custom, dict), "Result must be a dict"
    assert "text" in result_custom, "Result must have 'text'"
    print("[Test 1 PASSED]")

    print("\n[Test 2] Testing Request -> Process -> Response (Valid audio)...")
    req_valid = Request.from_values(
        path="/plugins/_whisper_stt/transcribe",
        method="POST",
        data=json.dumps({"audio": audio_b64}),
        content_type="application/json",
    )
    proc_res = await handler.process({"audio": audio_b64}, req_valid)
    assert isinstance(proc_res, dict), "process() must return dict on success"
    assert proc_res.get("success") is True, f"process() failed: {proc_res}"
    assert "text" in proc_res, "process() must have text"
    
    resp_success = await handler.handle_request(req_valid)
    print("Result 2 (Response):", resp_success, f"Status: {resp_success.status_code}")
    assert isinstance(resp_success, Response), "handle_request must return Flask Response"
    assert resp_success.status_code == 200, f"Expected 200, got {resp_success.status_code}"
    assert resp_success.mimetype == "application/json", f"Expected application/json, got {resp_success.mimetype}"
    body_data = json.loads(resp_success.get_data(as_text=True))
    assert body_data.get("success") is True, f"Body success must be True: {body_data}"
    assert "text" in body_data, "Body must contain text"
    print("[Test 2 PASSED]")

    print("\n[Test 3] Testing Request -> Response (Missing audio edge case)...")
    req_missing = Request.from_values(
        path="/plugins/_whisper_stt/transcribe",
        method="POST",
        data=json.dumps({}),
        content_type="application/json",
    )
    resp_missing = await handler.handle_request(req_missing)
    print("Result 3 (Missing Audio Response):", resp_missing, f"Status: {resp_missing.status_code}, Body: {resp_missing.get_data(as_text=True)}")
    assert isinstance(resp_missing, Response), "Must return a Response object"
    assert resp_missing.status_code == 400, f"Expected status 400, got {resp_missing.status_code}"
    assert "Missing audio" in resp_missing.get_data(as_text=True), "Response body must report missing audio"
    print("[Test 3 PASSED]")

    print("\n[Test 4] Testing standard Whisper model (tiny)...")
    config_default = {
        "model_size": "tiny",
        "custom_model": "",
        "language": "en",
    }
    result_default = await runtime.transcribe(audio_b64, config=config_default)
    print("Result 4 (Standard Whisper tiny):", result_default)
    assert isinstance(result_default, dict), "Result must be a dict"
    assert "text" in result_default, "Result must have 'text'"
    print("[Test 4 PASSED]")

    print("\n==========================================================")
    print("=== ALL 4 TESTS (RUNTIME, REQUEST & RESPONSE) PASSED! ===")
    print("==========================================================")


if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except Exception as e:
        print(f"\n[TEST FAILED] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
