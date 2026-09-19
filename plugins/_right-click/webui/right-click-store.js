import { createStore } from "/js/AlpineStore.js";
import { callJsonApi } from "/js/api.js";
import { toastFrontendError } from "/components/notifications/notification-store.js";
import { store as chatInputStore } from "/components/chat/input/input-store.js";
import { store as sidebarStore } from "/components/sidebar/sidebar-store.js";
import { store as fileBrowserStore } from "/components/modals/file-browser/file-browser-store.js";

const model = {
  kind: "",
  chatId: "",
  menu: null,
  trigger: null,

  async open(event) {
    this.close();
    if (event.defaultPrevented || event.shiftKey) return;
    const target = event.target;
    if (target.closest('input:not([type="checkbox"]), textarea, [contenteditable="true"], [inert]')) return;
    const row = target.closest(".chat-container, .task-container, .file-item");
    if (!row) return;
    const isFile = row.matches(".file-item");
    if (isFile && fileBrowserStore.isPickerMode()) return;
    const trigger = row.querySelector(isFile ? ".dropdown-trigger" : '[aria-haspopup="menu"]');
    if (!trigger || trigger.disabled) return;

    event.preventDefault();
    sidebarStore.menuClose();
    sidebarStore.rowMenuClose();
    fileBrowserStore.closeDropdown();
    // Use the existing row action, including its owning Files host and plugin actions.
    trigger.click();
    this.trigger = trigger;
    this.kind = isFile ? "file" : "sidebar";
    this.chatId = !isFile && sidebarStore.rowMenuKind === "chat"
      ? sidebarStore.rowMenuOpenId.slice("chat:".length) : "";

    const rect = row.getBoundingClientRect();
    const x = event.clientX || rect.left;
    const y = event.clientY || rect.bottom;
    const anchor = { getBoundingClientRect: () => ({ left: x, right: x, top: y, bottom: y }) };
    const position = width => {
      const style = fileBrowserStore.getDropdownStyle(anchor, width, false);
      if (isFile) fileBrowserStore.dropdownStyle = style;
      else sidebarStore.rowMenuStyle = style;
    };
    position(180);
    await globalThis.Alpine.nextTick();
    if (this.trigger !== trigger || !this.isOpen()) return;
    this.menu = Array.from(document.querySelectorAll(isFile ? ".file-actions-menu" : ".sidebar-row-actions-menu"))
      .find(menu => menu.checkVisibility()) || null;
    if (this.menu) {
      position(this.menu.offsetWidth);
      if (!event.pointerType) this.buttons()[0]?.focus({ preventScroll: true });
    }
  },

  isOpen() {
    return this.kind === "file" ? !!fileBrowserStore.openDropdownPath
      : this.kind === "sidebar" && !!sidebarStore.rowMenuOpenId;
  },

  async openChatFiles() {
    const ctxid = this.chatId;
    this.close();
    if (!ctxid) return;
    try {
      const response = await callJsonApi("/plugins/_right-click/chat_files_path", { ctxid });
      if (!response?.ok || !response.path) throw new Error(response?.error || "Could not find the chat's files.");
      await chatInputStore.browseFiles(response.path);
    } catch (error) {
      void toastFrontendError(error?.message || "Could not open chat files.", "Open in Files");
    }
  },

  close(restoreFocus = false) {
    if (this.kind === "file") fileBrowserStore.closeDropdown();
    if (this.kind === "sidebar") sidebarStore.rowMenuClose();
    if (restoreFocus) this.trigger?.focus({ preventScroll: true });
    this.kind = "";
    this.chatId = "";
    this.menu = null;
    this.trigger = null;
  },

  closeOutside(event) {
    if (!this.menu?.contains(event.target)) this.close();
  },

  buttons() {
    return Array.from(this.menu?.querySelectorAll("button:not(:disabled)") || [])
      .filter(button => button.checkVisibility());
  },

  keydown(event) {
    if (!this.isOpen()) return;
    if (event.key === "Escape" || event.key === "Tab") {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
      }
      this.close(true);
    } else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      event.stopPropagation();
      const buttons = this.buttons();
      const index = buttons.indexOf(document.activeElement);
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1
        : event.key === "ArrowDown" ? (index + 1) % buttons.length
        : (index < 0 ? buttons.length - 1 : (index - 1 + buttons.length) % buttons.length);
      buttons[next]?.focus({ preventScroll: true });
    }
  },
};

export const store = createStore("rightClick", model);
