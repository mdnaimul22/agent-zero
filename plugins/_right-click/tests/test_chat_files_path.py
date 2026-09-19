import asyncio
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace


def test_saved_chat_folder_and_unsaved_fallback(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    module = import_module('plugins._right-click.api.chat_files_path')
    monkeypatch.setattr(module.persist_chat, 'CHATS_FOLDER', str(tmp_path))
    contexts = {'saved': SimpleNamespace(id='saved'), 'new': SimpleNamespace(id='new')}
    monkeypatch.setattr(module.AgentContext, 'get', contexts.get)
    (tmp_path / 'saved').mkdir()
    handler = module.ChatFilesPath(None, None)

    def resolve(ctxid):
        return asyncio.run(handler.process({'ctxid': ctxid}, None))

    assert resolve('saved') == {'ok': True, 'path': str(tmp_path / 'saved')}
    assert resolve('new') == {'ok': True, 'path': str(tmp_path)}
    assert not (tmp_path / 'new').exists(), 'browsing must not create or persist a chat'
    assert resolve('missing').status_code == 404
    assert resolve('../../etc').status_code == 404
    for ctxid in ('', None, []):
        assert resolve(ctxid).status_code == 400
    assert handler.requires_auth() is True
    assert handler.requires_csrf() is True
