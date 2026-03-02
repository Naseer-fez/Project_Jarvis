let ws;
let reconnectTimer = null;

function byId(id) {
  return document.getElementById(id);
}

function render(data) {
  const state = data.state || "OFFLINE";
  const badge = byId("state-badge");
  if (badge) {
    badge.textContent = state;
    badge.className = `state-badge state-${state}`;
  }

  const lastInput = byId("last-input");
  if (lastInput) lastInput.textContent = data.last_input || "";

  const response = byId("last-response");
  if (response) {
    response.textContent = data.last_response || "";
    response.scrollTop = response.scrollHeight;
  }

  const info = byId("session-info");
  if (info) {
    const ollama = data.ollama_online ? "●" : "○";
    info.textContent = `Session ${data.session_id || "n/a"} | Model ${data.model || "unknown"} | Uptime ${data.uptime_seconds || 0}s | Memory ${data.memory_count || 0} | Active goals ${data.active_goals || 0} | Ollama ${ollama}`;
  }
}

function reconnect() {
  render({ state: "OFFLINE" });
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectWs();
  }, 3000);
}

function connectWs() {
  ws = new WebSocket("ws://localhost:7070/ws");
  ws.onmessage = (event) => {
    try {
      render(JSON.parse(event.data));
    } catch (_err) {}
  };
  ws.onclose = reconnect;
  ws.onerror = reconnect;
}

document.addEventListener("DOMContentLoaded", () => {
  connectWs();

  const form = byId("command-form");
  if (!form) return;

  const input = byId("command-input");
  const spinner = byId("cmd-spinner");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = (input?.value || "").trim();
    if (!text) return;

    spinner?.classList.remove("hidden");
    try {
      await fetch("http://localhost:7070/command", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Dashboard-Token": "jarvis"
        },
        body: JSON.stringify({ text })
      });
      if (input) input.value = "";
    } finally {
      spinner?.classList.add("hidden");
    }
  });
});
