"""Exercise right-click routing with the real menu and chat stores."""
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def test_right_click_menus_and_targeted_export():
    def store_source(path):
        source = (ROOT / path).read_text()
        source = re.sub(r'^import\b[\s\S]*?;\n', '', source, flags=re.M)
        source = source.replace('export const store =', 'const store =')
        source = source.replace('export { store };', '')
        return '(() => {\n' + source + '\nreturn store;\n})()'

    script = r'''
import assert from 'node:assert/strict';
const createStore = (_name, model) => model;
const createFileTree = () => ({});
const window = { innerWidth: 800, innerHeight: 600 };
globalThis.Alpine = { nextTick: async () => {} };
const document = { activeElement: null, querySelectorAll: () => [menu] };
const buttons = [0, 1].map(id => ({
  id, checkVisibility: () => true, focus() { document.activeElement = this; },
}));
const menu = {
  offsetWidth: 210, checkVisibility: () => true,
  querySelectorAll: () => buttons, contains: target => target === menu || buttons.includes(target),
};
const exported = [];
const sendJsonData = async (url, body) => {
  assert.equal(url, '/chat_export'); exported.push(body.ctxid);
  return { ctxid: body.ctxid, content: '{}' };
};
const getContext = () => 'fallback';
const toast = () => {};
const toastFetchError = (_message, error) => { throw error; };
const fileRequests = [], openedPaths = [], errors = [];
let pathResponse = {ok:true, path:'/a0/usr/chats/clicked'};
const callJsonApi = async (url, body) => {
  assert.equal(url, '/plugins/_right-click/chat_files_path'); fileRequests.push(body.ctxid);
  return pathResponse;
};
const chatInputStore = {browseFiles: async path => openedPaths.push(path)};
const toastFrontendError = message => errors.push(message);
'''
    for name, path in [
        ('sidebarStore', 'webui/components/sidebar/sidebar-store.js'),
        ('fileBrowserStore', 'webui/components/modals/file-browser/file-browser-store.js'),
        ('chatsStore', 'webui/components/sidebar/chats/chats-store.js'),
        ('rightClick', 'plugins/_right-click/webui/right-click-store.js'),
    ]:
        script += f'const {name} = {store_source(path)};\n'

    script += r'''
const owner = {};
let currentKind = 'chat', currentId = 'unselected', blocked = false;
const rect = { left: 10, right: 230, top: 40, bottom: 80 };
const trigger = {
  getBoundingClientRect: () => rect, closest: () => owner,
  focus() { document.activeElement = this; },
  click() {
    rightClick.closeOutside({ target: this });
    if (currentKind === 'file') fileBrowserStore.toggleDropdown(currentId, this);
    else sidebarStore.rowMenuToggle(`${currentKind}:${currentId}`, currentKind, this);
  },
};
const row = {
  matches: () => currentKind === 'file', querySelector: () => trigger,
  getBoundingClientRect: () => rect,
};
const event = (extra = {}) => ({
  target: { closest: selector => selector.startsWith('input') ? blocked : row },
  clientX: 790, clientY: 590, defaultPrevented: false,
  preventDefault() { this.defaultPrevented = true; }, ...extra,
});
chatsStore.selected = 'selected';
chatsStore.downloadFile = () => {};
for (const [kind, id] of [['chat', 'unselected'], ['chat', 'child'], ['task', 'scheduled'], ['file', '/a0/test.txt']]) {
  currentKind = kind; currentId = id;
  const opened = event();
  await rightClick.open(opened);
  assert.equal(opened.defaultPrevented, true);
  assert.equal(rightClick.isOpen(), true);
  assert.equal(chatsStore.selected, 'selected', 'right-click preserves active chat');
  assert.equal(document.activeElement, buttons[0]);
  const style = kind === 'file' ? fileBrowserStore.dropdownStyle : sidebarStore.rowMenuStyle;
  assert.equal(style.left, '582px', 'use measured width at right edge');
  assert.equal(style.top, 'auto', 'open upward at bottom edge');
  assert.equal(style.bottom, '16px');
  if (kind === 'file') assert.equal(fileBrowserStore.dropdownOwner, owner);
  if (kind === 'chat') {
    assert.equal(rightClick.chatId, id);
    await chatsStore.saveChat(rightClick.chatId);
    assert.equal(exported.at(-1), id, 'export clicked chat, not selection');
  } else assert.equal(rightClick.chatId, '');
  await rightClick.open(event({ clientX: 25, clientY: 50 }));
  assert.equal(rightClick.isOpen(), true, 'repeated right-click reopens instead of toggling');
  rightClick.closeOutside({ target: menu });
  assert.equal(rightClick.isOpen(), true, 'menu clicks and scrolling stay usable');
  const key = key => ({ key, preventDefault() {}, stopPropagation() {} });
  rightClick.keydown(key('ArrowDown'));
  assert.equal(document.activeElement, buttons[1]);
  rightClick.keydown(key('Home'));
  assert.equal(document.activeElement, buttons[0]);
  rightClick.keydown(key('Escape'));
  assert.equal(rightClick.isOpen(), false);
  assert.equal(document.activeElement, trigger);
}
document.activeElement = null;
await rightClick.open(event({ pointerType: 'mouse' }));
assert.equal(document.activeElement, null, 'pointer menus do not force a keyboard focus ring');
rightClick.keydown({ key: 'ArrowUp', preventDefault() {}, stopPropagation() {} });
assert.equal(document.activeElement, buttons.at(-1), 'ArrowUp starts at the last menu item');
for (const extra of [{ shiftKey: true }, { defaultPrevented: true }, { target: { closest: () => null } }]) {
  await rightClick.open(event(extra));
  assert.equal(rightClick.isOpen(), false, 'respect native or already-handled menus');
}
blocked = true;
let native = event(); await rightClick.open(native);
assert.equal(native.defaultPrevented, false, 'inert rows and text fields keep native behavior');
blocked = false;
fileBrowserStore.pickerMode = 'text-open';
native = event(); await rightClick.open(native);
assert.equal(native.defaultPrevented, false, 'picker does not expose hidden file actions');
fileBrowserStore.pickerMode = '';
await rightClick.open(event());
rightClick.closeOutside({ target: {} });
assert.equal(rightClick.isOpen(), false);
currentKind = 'chat';
await rightClick.open(event());
trigger.click();
assert.equal(rightClick.kind, '', 'ordinary overflow click hides the extra chat actions');
assert.equal(sidebarStore.rowMenuOpenId, `chat:${currentId}`);
rightClick.chatId = 'clicked';
rightClick.kind = 'sidebar';
await rightClick.openChatFiles();
assert.deepEqual(fileRequests, ['clicked']);
assert.deepEqual(openedPaths, ['/a0/usr/chats/clicked']);
assert.equal(chatsStore.selected, 'selected', 'Open in Files keeps the current chat selected');
assert.equal(sidebarStore.rowMenuOpenId, '');
rightClick.chatId = 'missing';
pathResponse = {ok:false, error:'Chat unavailable'};
await rightClick.openChatFiles();
assert.equal(openedPaths.length, 1, 'failed lookup does not open another chat folder');
assert.deepEqual(errors, ['Chat unavailable']);
await chatsStore.saveChat();
assert.equal(exported.at(-1), 'selected', 'header Save Chat keeps its default target');
chatsStore.selected = '';
await chatsStore.saveChat();
assert.equal(exported.at(-1), 'fallback');
chatsStore.readJsonFiles = async () => [];
await chatsStore.loadChats();
assert.equal(exported.at(-1), 'fallback', 'cancelling Load does not send an empty request');
'''
    subprocess.run(['node', '--input-type=module'], input=script, text=True,
                   check=True, timeout=15, cwd=ROOT)


def test_file_delete_uses_shared_two_click_confirmation():
    html = (ROOT / 'plugins/_right-click/extensions/webui/file-browser-actions-menu/file-actions.html').read_text()
    handler = re.search(r'@click\.stop="([^"]*\$confirmClick[^"]*)"', html).group(1)
    source = (ROOT / 'webui/js/confirmClick.js').read_text()
    source = re.sub(r'^import\b[\s\S]*?;\n', '', source, flags=re.M)
    source = source.replace('export function ', 'function ')
    script = r'''
import assert from 'node:assert/strict';
const ICON_SELECTOR = 'x-icon';
const getIconName = icon => icon.name;
const setIconName = (icon, name) => icon.name = name;
let reset, deleted = [], closes = 0;
const setTimeout = callback => {reset = callback; return 1;};
const clearTimeout = () => {};
const icon = {name:'delete', textContent:''};
const classes = new Set();
const button = {
  textContent:'Delete', innerHTML:'<x-icon name="delete"></x-icon><span>Delete</span>',
  querySelector: () => icon, classList:{add:name => classes.add(name), remove:name => classes.delete(name)},
};
const file = {path:'/a0/tmp/delete-check'};
const $store = {fileBrowser:{deleteFile: item => deleted.push(item.path)}, rightClick:{close:() => closes++}};
''' + source + '\nconst click = ($event, $confirmClick, $store, file) => {' + handler + r'''};
click({currentTarget:button}, confirmClick, $store, file);
assert.equal(deleted.length, 0);
assert.equal(closes, 0, 'first click must keep the menu open');
assert.equal(classes.has('confirming'), true);
assert.match(button.innerHTML, /Confirm/);
reset();
assert.equal(classes.has('confirming'), false);
click({currentTarget:button}, confirmClick, $store, file);
assert.equal(deleted.length, 0, 'expired confirmation needs a new first click');
click({currentTarget:button}, confirmClick, $store, file);
assert.deepEqual(deleted, [file.path]);
assert.equal(closes, 1);
'''
    subprocess.run(['node', '--input-type=module'], input=script, text=True,
                   check=True, timeout=15, cwd=ROOT)


def test_files_navigation_applies_the_path_to_canvas_and_modal():
    source = (ROOT / 'webui/components/chat/input/input-store.js').read_text()
    source = re.sub(r'^import\b[\s\S]*?;\n', '', source, flags=re.M)
    source = source.replace('export { store };', '')
    script = r'''
import assert from 'node:assert/strict';
const createStore = (_name, model) => model;
const calls = [];
let opened = true;
const shortcuts = {
  getCurrentContextId: () => 'selected',
  callJsonApi: async () => ({ok:true, path:'/workdir'}),
};
const openLatestSurface = async (_id, payload) => {calls.push(['surface', payload.path]); return opened;};
const fileBrowserStore = {
  navigateToFolder: async path => calls.push(['navigate', path]),
  open: async path => calls.push(['fallback', path]),
};
''' + source + r'''
await store.browseFiles('/a0/usr/chats/clicked');
assert.deepEqual(calls, [['surface','/a0/usr/chats/clicked'], ['navigate','/a0/usr/chats/clicked']]);
calls.length = 0;
await store.browseFiles();
assert.deepEqual(calls, [['surface','/workdir'], ['navigate','/workdir']]);
calls.length = 0;
opened = false;
await store.browseFiles('/a0/usr/chats/clicked');
assert.deepEqual(calls, [['surface','/a0/usr/chats/clicked'], ['fallback','/a0/usr/chats/clicked']]);
'''
    subprocess.run(['node', '--input-type=module'], input=script, text=True,
                   check=True, timeout=15, cwd=ROOT)
