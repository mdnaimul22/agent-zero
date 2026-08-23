import { store as contextWindowStore } from "/plugins/_context_window/webui/context-window-store.js";

const OVERRIDE_REVISION_KEY = "_model_config_override_revision";
let lastContextId = "";
let lastRevision = null;

export default async function refreshContextWindow(ctx) {
  const snapshot = ctx?.snapshot;
  const contextId = String(snapshot?.context || "");
  if (!contextId) {
    lastContextId = "";
    lastRevision = null;
    return;
  }

  const contexts = Array.isArray(snapshot?.contexts) ? snapshot.contexts : [];
  const active = contexts.find(item => item?.id === contextId) || null;
  const revision = active?.[OVERRIDE_REVISION_KEY] || null;
  if (contextId === lastContextId && revision === lastRevision) return;

  lastContextId = contextId;
  lastRevision = revision;
  await contextWindowStore.refresh(contextId);
}
