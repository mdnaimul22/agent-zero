import { createStore } from "/js/AlpineStore.js";
import { callJsonApi } from "/js/api.js";
import { store as chatsStore } from "/components/sidebar/chats/chats-store.js";
import { store as preferencesStore } from "/components/sidebar/bottom/preferences/preferences-store.js";

const API_PATH = "/plugins/_context_window/context_window";
const ROWS = [
  { key: "messages", label: "Messages", opacity: 1 },
  { key: "system_tools", label: "System tools", opacity: 0.88 },
  { key: "skills", label: "Skills", opacity: 0.76 },
  { key: "mcp_tools", label: "MCP tools", opacity: 0.64 },
  { key: "system_prompt", label: "System prompt", opacity: 0.52 },
  { key: "extras", label: "Extras", opacity: 0.4 },
];

preferencesStore.registerUiControlVisibility("contextWindowUsage", {
  mobile: true,
  desktop: true,
});

function formatTokens(value) {
  const amount = Math.max(Number(value) || 0, 0);
  for (const [size, suffix] of [[1_000_000, "M"], [1_000, "K"]]) {
    if (amount >= size) return `${(amount / size).toFixed(1).replace(/\.0$/, "")}${suffix}`;
  }
  return String(Math.round(amount));
}

function formatPercent(value) {
  const rounded = Math.round(Math.max(Number(value) || 0, 0) * 10) / 10;
  return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)}%`;
}

function buildUsage(data = {}) {
  const tokens = Math.max(Number(data.tokens) || 0, 0);
  const contextWindow = Math.max(Number(data.context_window) || 0, 0);
  const breakdown = data.usage && typeof data.usage === "object" ? data.usage : {};
  const percent = contextWindow > 0 ? (tokens / contextWindow) * 100 : 0;
  const rows = ROWS.map(row => {
    const rowTokens = Math.max(Number(breakdown[row.key]) || 0, 0);
    const rowPercent = contextWindow > 0 ? (rowTokens / contextWindow) * 100 : 0;
    return {
      ...row,
      tokensLabel: formatTokens(rowTokens),
      percentLabel: formatPercent(rowPercent),
      dotStyle: `opacity:${row.opacity}`,
    };
  });
  const hasBreakdown = rows.some(row => Number(breakdown[row.key]) > 0);
  if (hasBreakdown) {
    const freeTokens = Math.max(contextWindow - tokens, 0);
    const freePercent = contextWindow > 0 ? (freeTokens / contextWindow) * 100 : 0;
    rows.push({
      key: "free_space",
      label: "Free space",
      tokensLabel: formatTokens(freeTokens),
      percentLabel: formatPercent(freePercent),
      dotStyle: "opacity:0.24",
    });
  }
  const percentLabel = formatPercent(percent);
  return {
    rows: hasBreakdown ? rows : [],
    hasBreakdown,
    missingBreakdown: !hasBreakdown,
    ariaLabel: `Context window ${percentLabel} used`,
    ringLabel: contextWindow ? `${Math.round(percent)}%` : "–",
    ringDasharray: `${Math.min(percent, 100)} 100`,
    summary: `${formatTokens(tokens)}/${contextWindow ? formatTokens(contextWindow) : "–"} (${percentLabel})`,
    meterStyle: `width:${Math.min(percent, 100)}%`,
  };
}

const model = {
  usage: buildUsage(),
  loadSeq: 0,
  open: false,

  get contextId() {
    return chatsStore?.getSelectedChatId?.() || globalThis.getContext?.() || "";
  },

  async onMount(watch) {
    await this.refresh();
    watch("$store.chats.selected", value => this.refresh(value || ""));
    watch("$store.chats.selectedContext?.running", (running, previous) => {
      if (previous && !running) void this.refresh();
    });
  },

  cleanup() {
    this.open = false;
    this.loadSeq += 1;
  },

  toggle() {
    this.open = !this.open;
    if (this.open) void this.refresh();
  },

  async refresh(contextId = this.contextId) {
    const requestSeq = ++this.loadSeq;
    if (!contextId) {
      this.usage = buildUsage();
      return this.usage;
    }
    try {
      const data = await callJsonApi(API_PATH, { context: contextId });
      if (requestSeq === this.loadSeq) this.usage = buildUsage(data);
    } catch (error) {
      if (requestSeq === this.loadSeq) console.error("Context window load failed:", error);
    }
    return this.usage;
  },
};

export const store = createStore("contextWindow", model);
