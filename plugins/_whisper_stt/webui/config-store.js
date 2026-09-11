import { callJsonApi, fetchApi } from "/js/api.js";
import { toastFrontendSuccess, toastFrontendError } from "/components/notifications/notification-store.js";

const STANDARD_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"];

window.createWhisperConfigModel = (context, config) => ({
  downloading: false,
  downloadedModels: [],
  standardModels: STANDARD_MODELS,
  dropdownOpen: false,
  confirmingDelete: null,
  confirmTimeout: null,

  downloadAbortController: null,

  downloadState: {
    active: false,
    stage: "",
    file: "",
    percent: 0,
    downloaded_mb: 0,
    total_mb: 0,
    speed_mb: 0,
    error: null,
    done: false,
  },

  resetDownloadState() {
    if (this.downloadAbortController) {
      try {
        this.downloadAbortController.abort();
      } catch (e) {}
      this.downloadAbortController = null;
    }
    this.downloadState = {
      active: false,
      stage: "",
      file: "",
      percent: 0,
      downloaded_mb: 0,
      total_mb: 0,
      speed_mb: 0,
      error: null,
      done: false,
    };
  },

  async cancelDownload() {
    const modelToCancel = (config?.custom_model || "").trim();
    if (this.downloadAbortController) {
      try {
        this.downloadAbortController.abort();
      } catch (e) {}
      this.downloadAbortController = null;
    }
    try {
      await callJsonApi("/plugins/_whisper_stt/cancel_download", { model_name: modelToCancel });
    } catch (e) {}
    this.downloading = false;
    this.downloadState.active = false;
    this.downloadState.error = "Download cancelled by user.";
    void toastFrontendError("Download cancelled.", "Whisper STT");
  },

  isCustomModel(name) {
    return !!name && !this.standardModels.includes(name.toLowerCase());
  },

  truncateModelName(name) {
    if (!name) return "Base";
    const lower = name.toLowerCase();
    if (this.standardModels.includes(lower)) {
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    }
    return name.length > 20 ? name.slice(0, 20) + "…" : name;
  },

  formatStreamStage(text) {
    if (!text) return "";
    return text.length > 28 ? text.slice(0, 26) + "…" : text;
  },

  openDropdown() {
    this.dropdownOpen = true;
  },

  closeDropdown() {
    this.dropdownOpen = false;
    this.confirmingDelete = null;
    if (this.confirmTimeout) {
      clearTimeout(this.confirmTimeout);
      this.confirmTimeout = null;
    }
  },

  toggleDropdown() {
    if (this.dropdownOpen) {
      this.closeDropdown();
    } else {
      this.openDropdown();
    }
  },

  selectModel(m) {
    if (!config) return;
    config.model_size = m;
    this.closeDropdown();
    void toastFrontendSuccess(`Selected model: ${m}`, "Whisper STT");
  },

  handleDeleteClick(m, event) {
    if (event) event.stopPropagation();
    if (this.confirmingDelete === m) {
      this.confirmingDelete = null;
      if (this.confirmTimeout) {
        clearTimeout(this.confirmTimeout);
        this.confirmTimeout = null;
      }
      this.deleteModel(m);
    } else {
      this.confirmingDelete = m;
      if (this.confirmTimeout) clearTimeout(this.confirmTimeout);
      this.confirmTimeout = setTimeout(() => {
        if (this.confirmingDelete === m) {
          this.confirmingDelete = null;
        }
      }, 3500);
    }
  },

  async init() {
    try {
      const res = await callJsonApi("/plugins/_whisper_stt/status", {});
      if (res && Array.isArray(res.downloaded_models)) {
        this.downloadedModels = res.downloaded_models;
      }
      if (this.isCustomModel(config?.model_size) && !this.downloadedModels.includes(config.model_size)) {
        this.downloadedModels.push(config.model_size);
      }
      if (res && res.download_progress && res.download_progress.type === "progress") {
        this.downloadState.active = true;
        this.downloadState.percent = res.download_progress.percent || 0;
        this.downloadState.downloaded_mb = res.download_progress.downloaded_mb || 0;
        this.downloadState.total_mb = res.download_progress.total_mb || 0;
        this.downloadState.speed_mb = res.download_progress.speed_mb || 0;
        this.downloadState.stage = res.download_progress.stage || "";
      }
    } catch (e) {}
  },

  async deleteModel(modelToDelete) {
    if (!modelToDelete || !this.isCustomModel(modelToDelete)) return;

    try {
      const res = await callJsonApi("/plugins/_whisper_stt/delete_model", { model_name: modelToDelete });
      if (res && res.success) {
        this.downloadedModels = this.downloadedModels.filter((m) => m !== modelToDelete);
        if (config && config.model_size === modelToDelete) {
          config.model_size = "tiny";
        }
        void toastFrontendSuccess(`Model "${modelToDelete}" deleted from disk. Tiny selected as default.`, "Whisper STT");
      } else {
        void toastFrontendError(res?.error || "Failed to delete model", "Whisper STT");
      }
    } catch (err) {
      void toastFrontendError(err?.message || "Error deleting model", "Whisper STT");
    } finally {
      this.closeDropdown();
    }
  },

  async downloadModel() {
    const modelToLoad = (config?.custom_model || "").trim();
    if (!modelToLoad) {
      void toastFrontendError("Please enter a custom model name to download.", "Whisper STT");
      return;
    }
    this.downloading = true;
    this.downloadAbortController = new AbortController();
    this.downloadState = {
      active: true,
      stage: `Verifying model source: ${modelToLoad}...`,
      file: "",
      percent: 1,
      downloaded_mb: 0,
      total_mb: 0,
      speed_mb: 0,
      error: null,
      done: false,
    };

    try {
      const response = await fetchApi("/plugins/_whisper_stt/download_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelToLoad }),
        signal: this.downloadAbortController.signal,
      });

      if (!response.ok) {
        const txt = await response.text();
        throw new Error(txt || "Download failed, please check model source.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const ev = JSON.parse(line);
            if (ev.type === "start") {
              this.downloadState.total_mb = ev.total_mb || 0;
              this.downloadState.stage = `Starting download (${ev.files_count} files)...`;
            } else if (ev.type === "progress") {
              this.downloadState.percent = ev.percent;
              this.downloadState.downloaded_mb = ev.downloaded_mb;
              this.downloadState.total_mb = ev.total_mb;
              this.downloadState.speed_mb = ev.speed_mb;
              this.downloadState.stage = ev.stage || `Downloading: ${ev.file}`;
            } else if (ev.type === "error") {
              this.downloadState.active = false;
              this.downloadState.error = ev.message || "Download failed, please check model source.";
              void toastFrontendError(this.downloadState.error, "Whisper STT");
              return;
            } else if (ev.type === "done") {
              this.downloadState.active = false;
              this.downloadState.done = true;
              this.downloadState.percent = 100;
              this.downloadState.stage = `Model download complete: ${modelToLoad}`;
              if (!this.downloadedModels.includes(modelToLoad)) {
                this.downloadedModels.push(modelToLoad);
              }
              if (config) {
                config.model_size = modelToLoad;
                config.custom_model = "";
              }
              void toastFrontendSuccess(`Model "${modelToLoad}" downloaded successfully.`, "Whisper STT");
              return;
            }
          } catch (pe) {}
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        this.downloadState.active = false;
        this.downloadState.error = "Download cancelled by user.";
        return;
      }
      this.downloadState.active = false;
      this.downloadState.error = err.message || "Download failed, please check model source.";
      void toastFrontendError(this.downloadState.error, "Whisper STT");
    } finally {
      this.downloading = false;
      this.downloadAbortController = null;
    }
  },
});
