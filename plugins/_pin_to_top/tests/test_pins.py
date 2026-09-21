import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from plugins._pin_to_top.api.get_pins import GetPins
from plugins._pin_to_top.api.toggle_pin import TogglePin
from plugins._pin_to_top.helpers import pins


@pytest.fixture()
def persistent_store(monkeypatch):
    values = {}
    monkeypatch.setattr(
        pins.kvp,
        "get_persistent",
        lambda key, default=None: values.get(key, default),
    )
    monkeypatch.setattr(
        pins.kvp,
        "set_persistent",
        lambda key, value: values.__setitem__(key, value),
    )
    return values


def test_pins_are_persistent_and_separated_by_kind(monkeypatch, persistent_store):
    timestamps = iter((100.0, 200.0))
    monkeypatch.setattr(pins.time, "time", lambda: next(timestamps))

    assert pins.toggle_pin("chat", "chat-1") == (True, 100.0)
    assert pins.toggle_pin("task", "task-1") == (True, 200.0)
    assert pins.get_pins() == {
        "chat": {"chat-1": 100.0},
        "task": {"task-1": 200.0},
        "project": {},
    }

    assert pins.toggle_pin("chat", "chat-1") == (False, 0.0)
    assert pins.get_pins()["chat"] == {}


@pytest.mark.parametrize(
    ("kind", "item_id"),
    (("unknown", "item-1"), ("chat", ""), ("task", "x" * 513)),
)
def test_invalid_pin_input_is_rejected(kind, item_id, persistent_store):
    with pytest.raises(ValueError):
        pins.toggle_pin(kind, item_id)


def test_toggle_api_returns_a_bad_request_for_invalid_input(persistent_store):
    response = asyncio.run(
        TogglePin(None, None).process({"kind": "unknown", "item_id": "item-1"}, None)
    )

    assert response.status_code == 400


def test_project_pins_validate_projects_and_preserve_legacy_pins(
    monkeypatch, persistent_store, tmp_path
):
    persistent_store[pins.STORE_KEY] = {"chat": {"chat-1": 100.0}, "task": {"task-1": 200.0}}
    monkeypatch.setattr(pins.projects, "PROJECTS_PARENT_DIR", str(tmp_path))
    monkeypatch.setattr(pins.time, "time", lambda: 300.0)
    header = tmp_path / "my-project" / ".a0proj" / "project.json"
    header.parent.mkdir(parents=True)
    header.write_text("{}")

    assert pins.get_pins()["project"] == {}
    response = asyncio.run(
        TogglePin(None, None).process({"kind": "project", "item_id": "my-project"}, None)
    )
    assert response == {"ok": True, "pinned": True, "timestamp": 300.0}
    assert asyncio.run(GetPins(None, None).process({}, None))["pins"] == {
        "chat": {"chat-1": 100.0},
        "task": {"task-1": 200.0},
        "project": {"my-project": 300.0},
    }
    for name in ("../my-project", "missing-project"):
        response = asyncio.run(
            TogglePin(None, None).process({"kind": "project", "item_id": name}, None)
        )
        assert response.status_code == 400

    header.unlink()
    assert pins.toggle_pin("project", "my-project") == (False, 0.0)
    assert pins.get_pins()["project"] == {}


@pytest.mark.skipif(not shutil.which("node"), reason="node is required")
def test_project_pin_store_keeps_order_and_handles_arbitrary_project_names():
    script = """
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
const source = readFileSync(process.argv[1], 'utf8')
  .replace(/^import .*;$/gm, '')
  .replace('export const store = createStore("pinToTop", model);', 'globalThis.store = model;');
const calls = [];
const context = {
  callJsonApi: async (path, input) => {
    calls.push([path, input]);
    return { pinned: calls.length === 1, timestamp: 300 };
  },
  toastFrontendError: message => { throw new Error(message); },
};
runInNewContext(source, context);
const store = context.store;
const rows = [{ id: 'constructor' }, { id: '__proto__' }, { id: 'other' }];
assert.equal(store.isProjectPinned('__proto__'), false);
await store.toggleProjectPin('__proto__');
assert.equal(calls[0][0], '/plugins/_pin_to_top/toggle_pin');
assert.equal(calls[0][1].kind, 'project');
assert.equal(calls[0][1].item_id, '__proto__');
assert.equal(store.isProjectPinned('__proto__'), true);
assert.equal(store.sortItems('project', rows).map(item => item.id).join(','), '__proto__,constructor,other');
assert.equal(store.isPinned('chat', '__proto__'), false);
await store.toggleProjectPin('__proto__');
assert.equal(store.isProjectPinned('__proto__'), false);
assert.equal(store.sortItems('project', rows).map(item => item.id).join(','), 'constructor,__proto__,other');
await store.toggleFromMenu('task:task-1', 'task');
assert.equal(calls[2][1].kind, 'task');
assert.equal(calls[2][1].item_id, 'task-1');
"""
    store = Path(__file__).resolve().parents[1] / "webui" / "pin-to-top-store.js"
    subprocess.run(["node", "--input-type=module", "-e", script, str(store)], check=True)


def test_pin_endpoints_keep_default_auth_and_csrf_protection():
    for handler in (GetPins, TogglePin):
        assert handler.requires_auth() is True
        assert handler.requires_csrf() is True
