"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const conversation = $("conversation");
  const notice = $("notice");
  const input = $("prompt");
  const form = $("chatForm");
  const dialog = $("accessDialog");
  let config = null, accessKey = "", pending = false, history = [];

  function setNotice(message, error = false) {
    notice.textContent = message || "";
    notice.className = "notice" + (error ? " error" : "");
  }
  function state() {
    const usable = config && config.available && !pending;
    input.disabled = !usable;
    $("send").disabled = !usable || !input.value.trim();
    $("sidebarLed").classList.toggle("ready", !!usable);
    $("sidebarStatus").textContent = usable ? "Cloud bereit" : pending ? "NERAL antwortet …" : "Private Beta";
  }
  function unlock() {
    if (!config?.available) return;
    if (dialog.open) return;
    $("accessError").textContent = "";
    dialog.showModal();
    $("accessKey").focus();
  }
  function clearChat() {
    if (pending) return;
    history = [];
    const welcome = $("welcome");
    conversation.replaceChildren(welcome);
    welcome.hidden = false;
    input.value = "";
    state();
    input.focus();
  }
  function ensureMessages() {
    $("welcome").hidden = true;
    if (!conversation.querySelector(".message-list")) {
      const list = document.createElement("div");
      list.className = "message-list";
      list.id = "messageList";
      conversation.append(list);
    }
    return $("messageList");
  }
  function message(role, content, {error = false, meta = ""} = {}) {
    const list = ensureMessages();
    const row = document.createElement("article");
    row.className = "message " + role + (error ? " error" : "");
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "assistant" ? "N" : "●";
    const bodyWrapper = document.createElement("div");
    bodyWrapper.className = "message-content";
    if (role === "assistant") {
      const name = document.createElement("div");
      name.className = "message-name";
      name.textContent = "NERAL";
      bodyWrapper.append(name);
    }
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = content;
    bodyWrapper.append(body);
    if (meta) {
      const tag = document.createElement("div");
      tag.className = "message-meta";
      tag.textContent = meta;
      bodyWrapper.append(tag);
    }
    row.append(avatar, bodyWrapper);
    list.append(row);
    conversation.scrollTop = conversation.scrollHeight;
    return {row, body};
  }
  function boundedHistory() {
    const pairs = history.slice(-12);
    while (pairs.length > 0 && new TextEncoder().encode(pairs.map(x => x.content).join("")).length > 8500) {
      pairs.splice(0, 2);
    }
    return pairs;
  }
  async function send(prompt) {
    if (pending || !config?.available) return;
    if (!accessKey) return unlock();
    const text = prompt.trim();
    if (!text || text.length > 4000) return;
    input.value = "";
    pending = true;
    setNotice("");
    state();
    message("user", text);
    const waiting = message("assistant", "NERAL denkt nach …");
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 48000);
      let res;
      try {
        res = await fetch("/api/chat", {
          method: "POST", credentials: "same-origin",
          headers: {"Content-Type": "application/json", "X-Neral-Access": accessKey},
          body: JSON.stringify({message:text,history:boundedHistory()}),
          signal: controller.signal
        });
      } finally { clearTimeout(timeout); }
      const out = await res.json();
      if (!res.ok || typeof out.answer !== "string" || out.base_model !== "Qwen/Qwen3.5-9B") {
        if (res.status === 401) { accessKey = ""; unlock(); }
        throw new Error(out.error || "NERAL ist gerade nicht erreichbar.");
      }
      waiting.body.textContent = out.answer;
      const meta = document.createElement("div");
      meta.className = "message-meta";
      meta.textContent = "Qwen3.5-9B · Hugging Face Inference · Antworten können Fehler enthalten";
      waiting.body.parentElement.append(meta);
      history.push({role:"user",content:text},{role:"assistant",content:out.answer});
      history = boundedHistory();
    } catch (err) {
      waiting.row.classList.add("error");
      waiting.body.textContent = err.name === "AbortError" ? "NERAL hat zu lange gebraucht. Bitte versuche es erneut." : "NERAL konnte gerade nicht antworten.";
      setNotice("Die Antwort wurde nicht gespeichert. " + (err.message || ""),true);
      input.value = text;
    } finally {
      pending = false;
      state();
      conversation.scrollTop = conversation.scrollHeight;
    }
  }
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    send(input.value);
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 170) + "px";
    state();
  });
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  document.querySelectorAll("[data-prompt]").forEach(button => {
    button.addEventListener("click", () => {
      input.value = button.dataset.prompt || "";
      state();
      input.focus();
    });
  });
  $("newChat").addEventListener("click", clearChat);
  $("accessForm").addEventListener("submit", event => {
    if ($("saveAccess").value !== "unlock") return;
    event.preventDefault();
    const value = $("accessKey").value;
    if (value.length < 16) {
      $("accessError").textContent = "Bitte gib einen gültigen Einladungscode ein.";
      return;
    }
    accessKey = value;
    $("accessKey").value = "";
    dialog.close();
    input.focus();
  });
  (async function boot() {
    state();
    try {
      const response = await fetch("/api/config",{cache:"no-store"});
      if (!response.ok) throw new Error("Dienst nicht erreichbar");
      config = await response.json();
      if (!config.available) {
        setNotice("NERAL Cloud ist vorbereitet, aber noch nicht für echte Modellantworten freigeschaltet. Es wird keine Demoantwort erfunden.");
      } else {
        setNotice("NERAL Cloud ist bereit. Du benötigst einen Beta-Einladungscode.");
        unlock();
      }
    } catch (error) {
      setNotice("Die Cloud-Verbindung konnte nicht geprüft werden. Bitte lade die Seite neu.",true);
    } finally {state();}
  })();
})();
