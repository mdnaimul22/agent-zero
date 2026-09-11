from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import tempfile
import time
import warnings
import wave
from typing import Any

import numpy as np
import requests
import whisper

from helpers import files, plugins
from helpers.notification import (
    NotificationManager,
    NotificationPriority,
    NotificationType,
)
from helpers.print_style import PrintStyle
from plugins._whisper_stt.helpers import migration


warnings.filterwarnings("ignore", category=FutureWarning)


PLUGIN_NAME = "_whisper_stt"
DEFAULT_CONFIG = {
    "model_size": "base",
    "custom_model": "",
    "language": "en",
    "message_mode": "send",
    "silence_threshold": 0.3,
    "silence_duration": 1000,
    "waiting_timeout": 2000,
    "noise_reduction": False,
}
STANDARD_MODEL_SIZES = {"tiny", "base", "small", "medium", "large", "turbo"}
VALID_MESSAGE_MODES = {"send", "draft"}
REGISTRY_PATH = files.get_abs_path("/tmp/models/whisper/models.json")

_model = None
_model_name = ""
_model_type = "whisper"
is_updating_model = False
_current_download_status: dict[str, Any] = {}
_cancelled_downloads: set[str] = set()


def cancel_download(model_name: str = "") -> None:
    global _current_download_status
    clean = model_name.strip()
    if clean:
        _cancelled_downloads.add(clean)
    else:
        _cancelled_downloads.add("__all__")
    _current_download_status = {
        "type": "error",
        "model": clean,
        "message": "Download cancelled by user.",
        "cancelled": True,
    }


def is_download_cancelled(model_name: str) -> bool:
    clean = model_name.strip()
    return clean in _cancelled_downloads or "__all__" in _cancelled_downloads


def clear_download_cancellation(model_name: str) -> None:
    clean = model_name.strip()
    _cancelled_downloads.discard(clean)
    _cancelled_downloads.discard("__all__")


def load_registry() -> dict[str, Any]:
    if files.exists(REGISTRY_PATH):
        try:
            return json.loads(files.read_file(REGISTRY_PATH))
        except Exception:
            return {}
    return {}


def save_registry(data: dict[str, Any]) -> None:
    try:
        files.write_file(REGISTRY_PATH, json.dumps(data, indent=2))
    except Exception as e:
        PrintStyle.error(f"Failed to save STT models registry: {e}")


def get_current_download_progress() -> dict[str, Any]:
    return dict(_current_download_status)


def get_model_path(clean_id: str) -> str | None:
    clean_id = clean_id.strip()
    if not clean_id:
        return None
    whisper_dir = files.get_abs_path("/tmp/models/whisper")
    registry = load_registry()
    if clean_id in registry:
        p = registry[clean_id].get("path")
        if p and os.path.exists(p):
            if os.path.isfile(p) and os.path.getsize(p) > 1024 * 1024:
                return p
            if os.path.isdir(p):
                for cand in ("model.bin", "model.safetensors", "pytorch_model.bin"):
                    cf = os.path.join(p, cand)
                    if os.path.isfile(cf) and os.path.getsize(cf) > 1024 * 1024:
                        return p
    folder_name = f"models--{clean_id.replace('/', '--')}"
    target = os.path.join(whisper_dir, folder_name)
    if os.path.isdir(target):
        for cand in ("model.bin", "model.safetensors", "pytorch_model.bin"):
            cf = os.path.join(target, cand)
            if os.path.isfile(cf) and os.path.getsize(cf) > 1024 * 1024:
                return target
    direct_target = os.path.join(whisper_dir, clean_id)
    if os.path.isdir(direct_target):
        for cand in ("model.bin", "model.safetensors", "pytorch_model.bin"):
            cf = os.path.join(direct_target, cand)
            if os.path.isfile(cf) and os.path.getsize(cf) > 1024 * 1024:
                return direct_target
    pt_file = os.path.join(whisper_dir, f"{clean_id}.pt")
    if os.path.isfile(pt_file) and os.path.getsize(pt_file) > 1024 * 1024:
        return pt_file
    return None


def is_model_downloaded(clean_id: str) -> bool:
    return get_model_path(clean_id) is not None


def get_downloaded_models() -> list[str]:
    registry = load_registry()
    models = list(registry.keys())
    whisper_dir = files.get_abs_path("/tmp/models/whisper")
    if os.path.isdir(whisper_dir):
        for fname in os.listdir(whisper_dir):
            if fname.endswith(".pt"):
                name = fname[:-3]
                if name not in STANDARD_MODEL_SIZES and name not in models:
                    models.append(name)
            elif fname.startswith("models--"):
                parts = fname[len("models--"):].split("--")
                if len(parts) >= 2:
                    repo_id = f"{parts[0]}/{'--'.join(parts[1:])}"
                    if repo_id not in models and is_model_downloaded(repo_id):
                        models.append(repo_id)
    return sorted(models)


def register_downloaded_model(model_name: str, model_type: str = "whisper") -> None:
    clean = model_name.strip()
    if not clean or clean in STANDARD_MODEL_SIZES:
        return
    registry = load_registry()
    registry[clean] = {
        "id": clean,
        "name": clean,
        "type": model_type,
    }
    save_registry(registry)


def stream_download_model(repo_id: str):
    global _current_download_status
    clean_id = repo_id.strip()
    clear_download_cancellation(clean_id)
    if not clean_id:
        err = {"type": "error", "model": clean_id, "message": "Download failed, please check model repository source."}
        _current_download_status = err
        yield err
        return

    existing_path = get_model_path(clean_id)
    if existing_path:
        tot_mb = 50.0
        if os.path.isfile(existing_path):
            tot_mb = round(os.path.getsize(existing_path) / (1024 * 1024), 1)
        elif os.path.isdir(existing_path):
            for cand in ("model.bin", "model.safetensors", "pytorch_model.bin"):
                cf = os.path.join(existing_path, cand)
                if os.path.isfile(cf):
                    tot_mb = round(os.path.getsize(cf) / (1024 * 1024), 1)
                    break
        done = {
            "type": "done",
            "model": clean_id,
            "status": "ok",
            "percent": 100.0,
            "downloaded_mb": tot_mb,
            "total_mb": tot_mb,
            "path": existing_path,
            "already_downloaded": True,
        }
        _current_download_status = done
        yield done
        return

    whisper_dir = files.get_abs_path("/tmp/models/whisper")
    os.makedirs(whisper_dir, exist_ok=True)
    folder_name = f"models--{clean_id.replace('/', '--')}"
    target_dir = os.path.join(whisper_dir, folder_name)

    session = requests.Session()
    session.headers.update({"User-Agent": "agent-zero-whisper-downloader"})

    if "/" not in clean_id and clean_id.lower() in STANDARD_MODEL_SIZES:
        model_key = clean_id.lower()
        model_url = getattr(whisper, "_MODELS", {}).get(model_key)
        if not model_url:
            err = {"type": "error", "model": clean_id, "message": "Download failed, please check model repository source."}
            _current_download_status = err
            yield err
            return

        dest_file = os.path.join(whisper_dir, f"{model_key}.pt")
        temp_file = dest_file + ".part"

        total_bytes = 0
        try:
            h_resp = session.head(model_url, timeout=10, allow_redirects=True)
            if h_resp.status_code == 200 and h_resp.headers.get("Content-Length"):
                total_bytes = int(h_resp.headers["Content-Length"])
        except Exception:
            pass
        if total_bytes == 0:
            total_bytes = 150 * 1024 * 1024

        start_ev = {
            "type": "start",
            "model": clean_id,
            "total_mb": round(total_bytes / (1024 * 1024), 1),
            "files_count": 1,
        }
        _current_download_status = start_ev
        yield start_ev

        total_dl = 0
        start_time = time.time()
        last_report = 0.0

        try:
            with session.get(model_url, stream=True, allow_redirects=True, timeout=30) as resp:
                if resp.status_code != 200:
                    err = {"type": "error", "model": clean_id, "message": "Download failed, please check model repository source."}
                    _current_download_status = err
                    yield err
                    return
                with open(temp_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=262144):
                        if is_download_cancelled(clean_id):
                            clear_download_cancellation(clean_id)
                            if os.path.exists(temp_file):
                                try:
                                    os.remove(temp_file)
                                except Exception:
                                    pass
                            err = {"type": "error", "model": clean_id, "message": "Download cancelled by user.", "cancelled": True}
                            _current_download_status = err
                            yield err
                            return
                        if not chunk:
                            continue
                        f.write(chunk)
                        total_dl += len(chunk)
                        now = time.time()
                        if now - last_report >= 0.5 or last_report == 0.0:
                            elapsed = max(now - start_time, 0.001)
                            speed_mb = (total_dl / elapsed) / (1024 * 1024)
                            pct = min(round((total_dl / max(total_bytes, 1)) * 100, 1), 99.0)
                            prog = {
                                "type": "progress",
                                "model": clean_id,
                                "file": f"{model_key}.pt",
                                "percent": pct,
                                "downloaded_mb": round(total_dl / (1024 * 1024), 1),
                                "total_mb": round(total_bytes / (1024 * 1024), 1),
                                "speed_mb": round(speed_mb, 2),
                                "stage": f"Downloading: {model_key}.pt",
                            }
                            _current_download_status = prog
                            yield prog
                            last_report = now
            if os.path.exists(temp_file):
                os.replace(temp_file, dest_file)
            tot_mb = round(total_dl / (1024 * 1024), 1)
            done = {
                "type": "done",
                "model": clean_id,
                "status": "ok",
                "percent": 100.0,
                "downloaded_mb": tot_mb,
                "total_mb": tot_mb,
                "path": dest_file,
            }
            _current_download_status = done
            yield done
            return
        except Exception as e:
            err = {"type": "error", "model": clean_id, "message": f"Download failed, please check model source: {e}"}
            _current_download_status = err
            yield err
            return

    hf_api_url = f"https://huggingface.co/api/models/{clean_id}"
    api_res = None
    try:
        api_res = session.get(hf_api_url, timeout=10)
        if api_res.status_code == 404:
            err = {"type": "error", "model": clean_id, "message": "Download failed, please check model repository source."}
            _current_download_status = err
            yield err
            return
        elif api_res.status_code != 200:
            head_check = session.head(f"https://huggingface.co/{clean_id}/resolve/main/config.json", timeout=10, allow_redirects=True)
            if head_check.status_code == 404:
                err = {"type": "error", "model": clean_id, "message": "Download failed, please check model repository source."}
                _current_download_status = err
                yield err
                return
    except Exception as e:
        err = {"type": "error", "model": clean_id, "message": f"Download failed, please check model source: {e}"}
        _current_download_status = err
        yield err
        return

    os.makedirs(target_dir, exist_ok=True)
    base_url = f"https://huggingface.co/{clean_id}/resolve/main/"
    is_ct2 = False
    try:
        bin_check = session.head(base_url + "model.bin", timeout=10, allow_redirects=True)
        if bin_check.status_code == 200:
            is_ct2 = True
    except Exception:
        pass

    files_to_fetch = []
    if is_ct2:
        files_to_fetch = ["config.json", "vocabulary.txt", "tokenizer.json", "model.bin"]
        try:
            v_check = session.head(base_url + "vocabulary.txt", timeout=10, allow_redirects=True)
            if v_check.status_code != 200:
                files_to_fetch[1] = "vocabulary.json"
        except Exception:
            pass
    else:
        try:
            model_info = api_res.json() if api_res and api_res.status_code == 200 else {}
            siblings = [s.get("rfilename", "") for s in model_info.get("siblings", [])]
            preferred = [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "preprocessor_config.json",
                "model.safetensors",
                "pytorch_model.bin",
            ]
            files_to_fetch = [f for f in preferred if f in siblings]
            if not any(f in ("model.safetensors", "pytorch_model.bin") for f in files_to_fetch):
                for s in siblings:
                    if s.endswith(".safetensors") or s.endswith(".bin"):
                        files_to_fetch.append(s)
        except Exception:
            files_to_fetch = ["config.json", "tokenizer.json", "model.safetensors"]

    if not files_to_fetch:
        err = {"type": "error", "model": clean_id, "message": "Download failed, please check model repository source."}
        _current_download_status = err
        yield err
        return

    total_expected_bytes = 0
    for fname in files_to_fetch:
        try:
            head_resp = session.head(base_url + fname, timeout=10, allow_redirects=True)
            if head_resp.status_code == 200:
                clen = head_resp.headers.get("Content-Length")
                if clen:
                    total_expected_bytes += int(clen)
        except Exception:
            pass

    if total_expected_bytes == 0:
        total_expected_bytes = 150 * 1024 * 1024

    total_downloaded = 0
    start_time = time.time()
    last_report_time = 0.0

    start_ev = {
        "type": "start",
        "model": clean_id,
        "is_ct2": is_ct2,
        "total_mb": round(total_expected_bytes / (1024 * 1024), 1),
        "files_count": len(files_to_fetch),
    }
    _current_download_status = start_ev
    yield start_ev

    try:
        for fname in files_to_fetch:
            if is_download_cancelled(clean_id):
                clear_download_cancellation(clean_id)
                err = {"type": "error", "model": clean_id, "message": "Download cancelled by user.", "cancelled": True}
                _current_download_status = err
                yield err
                return

            dest = os.path.join(target_dir, fname)
            temp = os.path.join(target_dir, f"{fname}.part")

            if os.path.exists(dest) and os.path.getsize(dest) > 500:
                total_downloaded += os.path.getsize(dest)
                continue

            url = base_url + fname
            with session.get(url, stream=True, allow_redirects=True, timeout=30) as resp:
                if resp.status_code != 200:
                    if fname in ("model.bin", "model.safetensors", "config.json"):
                        err = {
                            "type": "error",
                            "model": clean_id,
                            "message": f"Download failed, please check model source: {fname} (HTTP {resp.status_code})"
                        }
                        _current_download_status = err
                        yield err
                        return
                    continue

                with open(temp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=262144):
                        if is_download_cancelled(clean_id):
                            clear_download_cancellation(clean_id)
                            if os.path.exists(temp):
                                try:
                                    os.remove(temp)
                                except Exception:
                                    pass
                            err = {"type": "error", "model": clean_id, "message": "Download cancelled by user.", "cancelled": True}
                            _current_download_status = err
                            yield err
                            return
                        if not chunk:
                            continue
                        f.write(chunk)
                        total_downloaded += len(chunk)

                        now = time.time()
                        if now - last_report_time >= 0.5 or last_report_time == 0.0:
                            elapsed = max(now - start_time, 0.001)
                            speed_mb = (total_downloaded / elapsed) / (1024 * 1024)
                            pct = min(round((total_downloaded / max(total_expected_bytes, 1)) * 100, 1), 99.0)
                            prog = {
                                "type": "progress",
                                "model": clean_id,
                                "file": fname,
                                "percent": pct,
                                "downloaded_mb": round(total_downloaded / (1024 * 1024), 1),
                                "total_mb": round(total_expected_bytes / (1024 * 1024), 1),
                                "speed_mb": round(speed_mb, 2),
                                "stage": f"Downloading: {fname}",
                            }
                            _current_download_status = prog
                            yield prog
                            last_report_time = now

            if os.path.exists(temp):
                os.replace(temp, dest)

        tot_mb = round(total_downloaded / (1024 * 1024), 1)
        registry = load_registry()
        registry[clean_id] = {
            "repo_id": clean_id,
            "name": clean_id,
            "path": target_dir,
            "folder": folder_name,
            "size_mb": tot_mb,
            "type": "ct2" if is_ct2 else "hf",
            "downloaded": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_registry(registry)

        done = {
            "type": "done",
            "model": clean_id,
            "status": "ok",
            "percent": 100.0,
            "downloaded_mb": tot_mb,
            "total_mb": tot_mb,
            "path": target_dir,
        }
        _current_download_status = done
        yield done
    except Exception as e:
        err = {
            "type": "error",
            "model": clean_id,
            "message": f"Download failed, please check model source: {e}"
        }
        _current_download_status = err
        yield err



async def delete_model(model_name: str) -> bool:
    return await asyncio.to_thread(_delete_model_sync, model_name)


def _delete_model_sync(model_name: str) -> bool:
    global _model, _model_name
    clean = model_name.strip()
    if not clean or clean in STANDARD_MODEL_SIZES:
        return False

    if _model_name == clean:
        _model = None
        _model_name = ""

    registry = load_registry()
    if clean in registry:
        del registry[clean]
        save_registry(registry)

    whisper_dir = files.get_abs_path("/tmp/models/whisper")
    pt_file = os.path.join(whisper_dir, f"{clean}.pt")
    if os.path.isfile(pt_file):
        try:
            os.remove(pt_file)
        except Exception as e:
            PrintStyle.error(f"Error removing {pt_file}: {e}")

    folder_name = f"models--{clean.replace('/', '--')}"
    target_dir = os.path.join(whisper_dir, folder_name)
    if os.path.isdir(target_dir):
        try:
            shutil.rmtree(target_dir)
        except Exception as e:
            PrintStyle.error(f"Error removing {target_dir}: {e}")

    direct_dir = os.path.join(whisper_dir, clean)
    if os.path.isdir(direct_dir):
        try:
            shutil.rmtree(direct_dir)
        except Exception as e:
            PrintStyle.error(f"Error removing {direct_dir}: {e}")

    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    hf_model_dir = os.path.join(hf_cache_dir, folder_name)
    if os.path.isdir(hf_model_dir):
        try:
            shutil.rmtree(hf_model_dir)
        except Exception as e:
            PrintStyle.error(f"Error removing {hf_model_dir}: {e}")

    PrintStyle.standard(f"Deleted STT model: {clean}")
    return True



def is_hf_model(model_name: str) -> bool:
    name = model_name.strip()
    if "/" in name:
        return True
    if name.lower() in STANDARD_MODEL_SIZES:
        return False
    if os.path.isfile(name):
        return False
    return False



def _load_hf_model(model_name: str):
    import torch
    from transformers import pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_target = get_model_path(model_name) or model_name

    return pipeline(
        "automatic-speech-recognition",
        model=model_target,
        torch_dtype=torch_dtype,
        device=device,
    )


def _load_whisper_model(model_name: str):
    return whisper.load_model(
        name=model_name,
        download_root=files.get_abs_path("/tmp/models/whisper"),
    )


def normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(DEFAULT_CONFIG)
    if not isinstance(config, dict):
        return normalized

    model_size = str(config.get("model_size", normalized["model_size"]) or "").strip()
    if model_size:
        if (
            model_size in STANDARD_MODEL_SIZES
            or model_size in get_downloaded_models()
            or "/" in model_size
            or os.path.isfile(model_size)
        ):
            normalized["model_size"] = model_size

    custom_model = str(config.get("custom_model", normalized["custom_model"]) or "").strip()
    normalized["custom_model"] = custom_model

    language = str(config.get("language", normalized["language"]) or "").strip()
    if language:
        normalized["language"] = language

    message_mode = (
        str(config.get("message_mode", normalized["message_mode"]) or "")
        .strip()
        .lower()
    )
    if message_mode in VALID_MESSAGE_MODES:
        normalized["message_mode"] = message_mode

    try:
        silence_threshold = float(
            config.get("silence_threshold", normalized["silence_threshold"])
        )
        normalized["silence_threshold"] = min(max(silence_threshold, 0.0), 1.0)
    except (TypeError, ValueError):
        pass

    try:
        silence_duration = int(
            config.get("silence_duration", normalized["silence_duration"])
        )
        if silence_duration > 0:
            normalized["silence_duration"] = silence_duration
    except (TypeError, ValueError):
        pass

    try:
        waiting_timeout = int(config.get("waiting_timeout", normalized["waiting_timeout"]))
        if waiting_timeout > 0:
            normalized["waiting_timeout"] = waiting_timeout
    except (TypeError, ValueError):
        pass

    normalized["noise_reduction"] = bool(config.get("noise_reduction", normalized["noise_reduction"]))

    return normalized


def get_config() -> dict[str, Any]:
    migration.ensure_config_seeded()
    config = plugins.get_plugin_config(PLUGIN_NAME) or {}
    return normalize_config(config)


def get_loaded_model_name() -> str:
    return _model_name


def get_loaded_model_type() -> str:
    return _model_type


def is_globally_enabled() -> bool:
    return plugins.determined_toggle_from_paths(
        True, reversed(plugins.get_plugin_roots(PLUGIN_NAME))
    )


async def preload(model_name: str | None = None):
    cfg = get_config()
    resolved_model = str(model_name or cfg.get("custom_model") or cfg["model_size"]).strip()
    return await _preload(resolved_model)


async def _preload(model_name: str):
    global _model, _model_name, _model_type, is_updating_model

    while is_updating_model:
        await asyncio.sleep(0.1)

    try:
        is_updating_model = True
        if not _model or _model_name != model_name:
            NotificationManager.send_notification(
                NotificationType.INFO,
                NotificationPriority.NORMAL,
                f"Loading Whisper model: {model_name}...",
                display_time=99,
                group="whisper-preload",
            )
            PrintStyle.standard(f"Loading Whisper STT model: {model_name}")

            if is_hf_model(model_name):
                _model = await asyncio.to_thread(_load_hf_model, model_name)
                _model_type = "hf"
            else:
                _model = await asyncio.to_thread(_load_whisper_model, model_name)
                _model_type = "whisper"

            _model_name = model_name
            register_downloaded_model(model_name, _model_type)
            NotificationManager.send_notification(
                NotificationType.INFO,
                NotificationPriority.NORMAL,
                f"STT model '{model_name}' loaded.",
                display_time=3,
                group="whisper-preload",
            )
            PrintStyle.success(f"STT model loaded: {model_name} ({_model_type})")
    finally:
        is_updating_model = False



async def is_downloading() -> bool:
    return is_updating_model


async def is_downloaded() -> bool:
    return _model is not None


async def transcribe(
    audio_bytes_b64: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    cfg = normalize_config(config or get_config())
    model_name = str(cfg.get("custom_model") or cfg["model_size"]).strip()
    return await _transcribe(
        model_name,
        audio_bytes_b64,
        language=_resolve_language(str(cfg["language"])),
        noise_reduction=bool(cfg.get("noise_reduction", False)),
    )


def _apply_noise_reduction(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sample_rate, stationary=True, prop_decrease=0.75)
    except Exception as e:
        PrintStyle.error(f"Noise reduction failed: {e}")
        return audio


_hf_supports_language: dict[str, bool] = {}


def _decode_audio(audio_bytes: bytes, temp_path: str) -> np.ndarray:
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]:
        try:
            with io.BytesIO(audio_bytes) as bio:
                with wave.open(bio, "rb") as wf:
                    if wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2:
                        frames = wf.readframes(wf.getnframes())
                        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            pass
    return whisper.load_audio(temp_path)


async def _transcribe(
    model_name: str,
    audio_bytes_b64: str,
    *,
    language: str | None = None,
    noise_reduction: bool = False,
) -> dict[str, Any]:
    if not _model or _model_name != model_name:
        await _preload(model_name)

    audio_bytes = base64.b64decode(audio_bytes_b64)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(audio_bytes)
        temp_path = audio_file.name

    clean_audio = None
    if noise_reduction:
        try:
            raw_audio = await asyncio.to_thread(_decode_audio, audio_bytes, temp_path)
            clean_audio = await asyncio.to_thread(_apply_noise_reduction, raw_audio, 16000)
            int_data = np.clip(clean_audio * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(int_data.tobytes())
        except Exception as e:
            PrintStyle.error(f"Noise reduction processing error: {e}")
            clean_audio = None

    try:
        import torch

        if _model_type == "hf":
            if clean_audio is not None:
                audio_input = clean_audio
            else:
                audio_input = await asyncio.to_thread(_decode_audio, audio_bytes, temp_path)

            def _run_hf():
                with torch.inference_mode():
                    use_lang = _hf_supports_language.get(_model_name, True)
                    if language and use_lang:
                        try:
                            res = _model(audio_input, generate_kwargs={"language": language})
                            _hf_supports_language[_model_name] = True
                            return res
                        except Exception:
                            _hf_supports_language[_model_name] = False
                    return _model(audio_input)

            result = await asyncio.to_thread(_run_hf)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return {"text": text.strip(), "language": language or ""}
        else:
            use_fp16 = torch.cuda.is_available() and hasattr(_model, "device") and str(_model.device).startswith("cuda")
            kwargs: dict[str, Any] = {"fp16": use_fp16}
            if language:
                kwargs["language"] = language

            def _run_whisper():
                with torch.inference_mode():
                    return _model.transcribe(temp_path, **kwargs)

            result = await asyncio.to_thread(_run_whisper)
            return result if isinstance(result, dict) else {}
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def _resolve_language(language: str) -> str | None:
    value = language.strip().lower()
    if not value or value == "auto":
        return None
    return value

