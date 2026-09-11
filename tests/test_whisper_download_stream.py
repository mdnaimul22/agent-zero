import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins._whisper_stt.helpers import runtime
from plugins._whisper_stt.api.download_stream import DownloadStream
from plugins._whisper_stt.api.cancel_download import CancelDownload
from helpers.api import Request


def test_invalid_model_download_stream():
    events = list(runtime.stream_download_model("invalid-nonexistent-user-12345/does-not-exist"))
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "Download failed, please check model repository source." in events[0]["message"]


def test_empty_model_name():
    events = list(runtime.stream_download_model("   "))
    assert len(events) == 1
    assert events[0]["type"] == "error"


def test_valid_repo_stream_metadata():
    stream = runtime.stream_download_model("openai/whisper-tiny")
    first_ev = next(stream)
    assert first_ev["type"] in ("start", "done")
    if first_ev["type"] == "start":
        assert first_ev["total_mb"] > 0
        second_ev = next(stream)
        assert second_ev["type"] in ("progress", "done")
    stream.close()


async def test_api_handler_stream():
    handler = DownloadStream(app=None, thread_lock=threading.Lock())
    resp = await handler.process({"model_name": "invalid-nonexistent/test-xyz"}, request=None)
    assert resp.status_code == 200
    assert resp.mimetype == "application/x-ndjson"
    gen = resp.response
    first_line = next(gen)
    assert "Download failed, please check model repository source." in first_line
    if hasattr(gen, "close"):
        gen.close()


async def test_full_handle_request_lifecycle():
    handler = DownloadStream(app=None, thread_lock=threading.Lock())
    req = Request.from_values(
        path="/plugins/_whisper_stt/download_stream",
        method="POST",
        data='{"model_name": "invalid-nonexistent/test-xyz"}',
        content_type="application/json",
    )
    resp = await handler.handle_request(req)
    assert resp.status_code == 200
    assert resp.mimetype == "application/x-ndjson"
    gen = resp.response
    first_line = next(gen)
    assert "Download failed, please check model repository source." in first_line
    if hasattr(gen, "close"):
        gen.close()


async def test_cancel_download_api():
    handler = CancelDownload(app=None, thread_lock=threading.Lock())
    req = Request.from_values(
        path="/plugins/_whisper_stt/cancel_download",
        method="POST",
        data='{"model_name": "openai/whisper-base"}',
        content_type="application/json",
    )
    resp = await handler.handle_request(req)
    assert resp.status_code == 200
    assert resp.mimetype == "application/json"
    assert runtime.is_download_cancelled("openai/whisper-base")
    runtime.clear_download_cancellation("openai/whisper-base")
    assert not runtime.is_download_cancelled("openai/whisper-base")


if __name__ == "__main__":
    print("Running test_invalid_model_download_stream...", flush=True)
    test_invalid_model_download_stream()
    print("✓ test_invalid_model_download_stream passed!", flush=True)

    print("Running test_empty_model_name...", flush=True)
    test_empty_model_name()
    print("✓ test_empty_model_name passed!", flush=True)

    print("Running test_valid_repo_stream_metadata...", flush=True)
    test_valid_repo_stream_metadata()
    print("✓ test_valid_repo_stream_metadata passed!", flush=True)

    print("Running test_api_handler_stream...", flush=True)
    asyncio.run(test_api_handler_stream())
    print("✓ test_api_handler_stream passed!", flush=True)

    print("Running test_full_handle_request_lifecycle...", flush=True)
    asyncio.run(test_full_handle_request_lifecycle())
    print("✓ test_full_handle_request_lifecycle passed!", flush=True)

    print("Running test_cancel_download_api...", flush=True)
    asyncio.run(test_cancel_download_api())
    print("✓ test_cancel_download_api passed!", flush=True)

    print("\nALL 6 TESTS PASSED SUCCESSFULLY!", flush=True)
