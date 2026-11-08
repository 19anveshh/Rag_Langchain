(() => {
  "use strict";

  const STORE_KEY = "ragchat.threads.v1";
  const CURRENT_KEY = "ragchat.current";
  const MARKED_OK = typeof marked !== "undefined";
  const PURIFY_OK = typeof DOMPurify !== "undefined";

  const els = {
    sidebar: document.getElementById("sidebar"),
    scrim: document.getElementById("scrim"),
    menuBtn: document.getElementById("menuBtn"),
    newChatBtn: document.getElementById("newChatBtn"),
    threadList: document.getElementById("threadList"),
    threadTitle: document.getElementById("threadTitle"),
    container: document.getElementById("messagesContainer"),
    messages: document.getElementById("messages"),
    input: document.getElementById("input"),
    sendBtn: document.getElementById("sendBtn"),
  };

  const SUGGESTIONS = [
    "What topics are covered in the question bank?",
    "Summarize the key APIs from the documentation",
    "List the chapters in the DE question bank",
  ];

  let threads = loadThreads();
  let currentId = null;
  let streaming = false;
  let controller = null;
  let pacer = null;
  const turnText = new WeakMap();

  /* ---------- storage ---------- */

  function loadThreads() {
    try {
      const v = JSON.parse(localStorage.getItem(STORE_KEY));
      return Array.isArray(v) ? v : [];
    } catch {
      return [];
    }
  }

  function saveThreads() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(threads));
    } catch {}
  }

  function getThread(id) {
    return threads.find((t) => t.id === id) || null;
  }

  function setCurrent(id) {
    currentId = id;
    try {
      if (id) localStorage.setItem(CURRENT_KEY, id);
      else localStorage.removeItem(CURRENT_KEY);
    } catch {}
  }

  /* ---------- markdown ---------- */

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  if (MARKED_OK) {
    marked.use({ gfm: true, breaks: true });
    marked.use({
      renderer: {
        code({ text, lang }) {
          const language = (lang || "").trim().split(/\s+/)[0];
          let body = null;
          if (window.hljs && language && hljs.getLanguage(language)) {
            try { body = hljs.highlight(text, { language }).value; } catch {}
          }
          if (body == null && window.hljs) {
            try { body = hljs.highlightAuto(text).value; } catch {}
          }
          if (body == null) body = escapeHtml(text);
          return (
            `<div class="codeblock"><div class="code-head">` +
            `<span>${escapeHtml(language || "code")}</span>` +
            `<button class="copy-code">Copy</button></div>` +
            `<pre><code class="hljs">${body}</code></pre></div>`
          );
        },
      },
    });
  }

  function renderMD(text) {
    if (!MARKED_OK) return escapeHtml(text).replace(/\n/g, "<br>");
    let html;
    try {
      html = marked.parse(text);
    } catch {
      return `<p>${escapeHtml(text)}</p>`;
    }
    return PURIFY_OK ? DOMPurify.sanitize(html) : html;
  }

  /* ---------- reading-paced reveal ---------- */

  class Pacer {
    constructor(onTick, onDrained) {
      this.buf = "";
      this.pos = 0;
      this.done = false;
      this.timer = null;
      this.onTick = onTick;
      this.onDrained = onDrained;
    }
    append(t) {
      this.buf += t;
      if (!this.timer) this.timer = setInterval(() => this._tick(), 33);
    }
    _tick() {
      const ahead = this.buf.length - this.pos;
      const cps = ahead > 400 ? 140 : ahead > 150 ? 105 : 70;
      this.pos = Math.min(this.buf.length, this.pos + Math.max(1, Math.round(cps * 0.033)));
      this.onTick(this.buf.slice(0, this.pos));
      if (this.done && this.pos >= this.buf.length) this._drain();
    }
    finish() {
      this.done = true;
      if (this.pos >= this.buf.length) this._drain();
    }
    flush() {
      this.pos = this.buf.length;
      this.onTick(this.buf);
      this.done = true;
      this._drain();
    }
    _drain() {
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }
      this.onDrained(this.buf);
    }
  }

  /* ---------- assistant turn rendering ---------- */

  function createAssistantTurn() {
    const turn = document.createElement("div");
    turn.className = "turn-assistant";
    turn.innerHTML = `
      <div class="assistant-label"><span>RAG Agent</span><button class="copy-all">Copy</button></div>
      <div class="answer streaming"></div>`;
    els.messages.appendChild(turn);
    return {
      el: turn,
      finalized: false,
      stepsEl: null,
      stepsMeta: [],
      _answer: turn.querySelector(".answer"),
      _label: turn.querySelector(".assistant-label"),

      ensureSteps() {
        if (!this.stepsEl) {
          this.stepsEl = document.createElement("div");
          this.stepsEl.className = "steps";
          turn.insertBefore(this.stepsEl, this._label);
        }
        return this.stepsEl;
      },
      addStep(id, label) {
        this.stepsMeta.push({ id, label, detail: "" });
        const steps = this.ensureSteps();
        const row = document.createElement("div");
        row.className = "step-row";
        row.dataset.id = id;
        row.innerHTML = `<span class="spin"></span><span class="s-label"></span>`;
        row.querySelector(".s-label").textContent = label;
        row.addEventListener("click", () => row.classList.toggle("open"));
        const detail = document.createElement("div");
        detail.className = "step-detail";
        steps.appendChild(row);
        steps.appendChild(detail);
      },
      completeStep(id, detail) {
        const meta = this.stepsMeta.find((s) => s.id === id);
        if (meta) meta.detail = detail;
        const row = this.stepsEl && this.stepsEl.querySelector(`.step-row[data-id="${CSS.escape(id)}"]`);
        if (!row) return;
        row.querySelector(".spin").outerHTML =
          `<span class="done-icon">&#10003;</span>`;
        row.nextElementSibling.textContent = detail || "Done";
      },
      setAnswer(revealed) {
        if (this.finalized) return;
        this._answer.innerHTML = renderMD(revealed);
      },
      showError(msg) {
        const div = document.createElement("div");
        div.className = "turn-error";
        div.textContent = msg;
        turn.appendChild(div);
      },
      finalize(fullText) {
        if (this.finalized) return;
        this.finalized = true;
        this._answer.classList.remove("streaming");
        this._answer.innerHTML = fullText.trim() ? renderMD(fullText) : "";
        turnText.set(turn, fullText);
        if (this.stepsEl && this.stepsMeta.length) {
          this.stepsEl.classList.add("collapsed");
          const summary = document.createElement("button");
          summary.className = "steps-summary";
          const n = this.stepsMeta.length;
          summary.innerHTML = `<span></span><span class="chev">&#9662;</span>`;
          summary.firstChild.textContent = `${n} step${n > 1 ? "s" : ""}`;
          summary.addEventListener("click", () => this.stepsEl.classList.toggle("open"));
          this.stepsEl.insertBefore(summary, this.stepsEl.firstChild);
        }
      },
      showStopped() {
        const note = document.createElement("div");
        note.className = "stopped-note";
        note.textContent = "Stopped.";
        turn.appendChild(note);
      },
    };
  }

  function renderAssistantHistory(msg) {
    const turn = createAssistantTurn();
    turn.finalized = true;
    turn._answer.classList.remove("streaming");
    if (msg.steps && msg.steps.length) {
      turn.ensureSteps();
      for (const s of msg.steps) {
        const row = document.createElement("div");
        row.className = "step-row";
        row.innerHTML = `<span class="done-icon">&#10003;</span><span class="s-label"></span>`;
        row.querySelector(".s-label").textContent = s.label;
        row.addEventListener("click", () => row.classList.toggle("open"));
        const detail = document.createElement("div");
        detail.className = "step-detail";
        detail.textContent = s.detail || "Done";
        turn.stepsEl.appendChild(row);
        turn.stepsEl.appendChild(detail);
      }
      turn.stepsMeta = msg.steps.map((s) => ({ ...s }));
      turn.stepsEl.classList.add("collapsed");
      const n = msg.steps.length;
      const summary = document.createElement("button");
      summary.className = "steps-summary";
      summary.innerHTML = `<span></span><span class="chev">&#9662;</span>`;
      summary.firstChild.textContent = `${n} step${n > 1 ? "s" : ""}`;
      summary.addEventListener("click", () => turn.stepsEl.classList.toggle("open"));
      turn.stepsEl.insertBefore(summary, turn.stepsEl.firstChild);
    }
    turn._answer.innerHTML = msg.content.trim() ? renderMD(msg.content) : "";
    turnText.set(turn.el, msg.content);
  }

  /* ---------- scroll rules ---------- */

  function positionScrollOnce() {
    const c = els.container;
    const distFromBottom = c.scrollHeight - c.scrollTop - c.clientHeight;
    if (distFromBottom > c.clientHeight * 0.4) {
      c.scrollTo(0, c.scrollHeight);
    }
  }

  /* ---------- threads ---------- */

  function createThread() {
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
    threads.push({ id, title: "", createdAt: Date.now(), updatedAt: Date.now(), messages: [] });
    return id;
  }

  function timeAgo(ts) {
    const s = Math.max(1, Math.floor((Date.now() - ts) / 1000));
    if (s < 60) return "just now";
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    const d = Math.floor(h / 24);
    return d === 1 ? "yesterday" : `${d}d ago`;
  }

  function renderSidebar() {
    const list = [...threads].sort((a, b) => b.updatedAt - a.updatedAt);
    els.threadList.innerHTML = "";
    if (!list.length) {
      const empty = document.createElement("div");
      empty.className = "threads-empty";
      empty.textContent = "No conversations yet";
      els.threadList.appendChild(empty);
      return;
    }
    for (const t of list) {
      const item = document.createElement("div");
      item.className = "thread-item" + (t.id === currentId ? " active" : "");
      item.innerHTML = `<div class="ti-text"><span class="ti-title"></span><span class="when"></span></div><button class="del" aria-label="Delete chat">&times;</button>`;
      item.querySelector(".ti-title").textContent = t.title || "Untitled";
      item.querySelector(".when").textContent = timeAgo(t.updatedAt);
      item.addEventListener("click", (e) => {
        if (e.target.closest(".del")) return;
        switchThread(t.id);
      });
      item.querySelector(".del").addEventListener("click", () => deleteThread(t.id));
      els.threadList.appendChild(item);
    }
  }

  function switchThread(id) {
    if (id === currentId) return;
    if (streaming && controller) controller.abort();
    setCurrent(id);
    renderThread(getThread(id));
    renderSidebar();
    closeMobileSidebar();
    els.input.focus();
  }

  function deleteThread(id) {
    threads = threads.filter((t) => t.id !== id);
    saveThreads();
    if (id === currentId) {
      setCurrent(null);
      renderThread(null);
    }
    renderSidebar();
  }

  function newChat() {
    if (streaming && controller) controller.abort();
    setCurrent(null);
    renderThread(null);
    renderSidebar();
    closeMobileSidebar();
    els.input.focus();
  }

  /* ---------- hero / thread rendering ---------- */

  function renderHero() {
    const hero = document.createElement("div");
    hero.className = "hero";
    hero.innerHTML = `
      <div class="hero-badge">RAG Agent</div>
      <h1>Ask your documents</h1>
      <p>Answers are grounded in the indexed PDFs &mdash; a documentation agent searches chunks, cross-checks them with analysts, then replies.</p>`;
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const s of SUGGESTIONS) {
      const chip = document.createElement("button");
      chip.className = "chip";
      chip.textContent = s;
      chip.addEventListener("click", () => send(s));
      chips.appendChild(chip);
    }
    hero.appendChild(chips);
    els.messages.appendChild(hero);
  }

  function renderThread(thread) {
    els.messages.innerHTML = "";
    if (!thread || thread.messages.length === 0) {
      renderHero();
    } else {
      for (const m of thread.messages) {
        if (m.role === "user") {
          const div = document.createElement("div");
          div.className = "turn-user";
          div.textContent = m.content;
          els.messages.appendChild(div);
        } else {
          renderAssistantHistory(m);
        }
      }
      // Restore anchored at the last question, not pinned to the bottom.
      const lastUser = els.messages.querySelector(".turn-user:last-of-type");
      if (lastUser) {
        els.container.scrollTo(0, Math.max(0, lastUser.offsetTop - 12));
      } else {
        els.container.scrollTo(0, els.container.scrollHeight);
      }
    }
    els.threadTitle.textContent = thread && thread.title ? thread.title : "New chat";
  }

  /* ---------- streaming ---------- */

  function setStreamingUI(on) {
    els.sendBtn.classList.toggle("stopping", on);
    els.sendBtn.setAttribute("aria-label", on ? "Stop generating" : "Send message");
    updateSendBtn();
  }

  function updateSendBtn() {
    els.sendBtn.disabled = streaming ? false : els.input.value.trim().length === 0;
  }

  async function send(text) {
    const content = text.trim();
    if (!content || streaming) return;
    if (pacer && pacer.timer !== null) pacer.flush();

    if (!currentId) setCurrent(createThread());
    const thread = getThread(currentId);
    if (!thread) return;

    if (els.messages.querySelector(".hero")) els.messages.innerHTML = "";
    thread.messages.push({ role: "user", content });
    thread.updatedAt = Date.now();
    if (!thread.title) thread.title = content.length > 42 ? content.slice(0, 42) + "\u2026" : content;
    saveThreads();

    const userDiv = document.createElement("div");
    userDiv.className = "turn-user";
    userDiv.textContent = content;
    els.messages.appendChild(userDiv);
    els.threadTitle.textContent = thread.title;
    renderSidebar();
    positionScrollOnce();

    const turn = createAssistantTurn();
    streaming = true;
    setStreamingUI(true);
    controller = new AbortController();

    const thisPacer = new Pacer(
      (revealed) => turn.setAnswer(revealed),
      (full) => {
        finalizeTurn(turn, thread, full);
        if (pacer === thisPacer) pacer = null;
      },
    );
    pacer = thisPacer;

    const history = thread.messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: m.content }));

    let aborted = false;
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const raw = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const line of raw.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            let evt;
            try {
              evt = JSON.parse(line.slice(6));
            } catch {
              continue;
            }
            if (evt.type === "step") turn.addStep(evt.id, evt.label);
            else if (evt.type === "step_done") turn.completeStep(evt.id, evt.detail);
            else if (evt.type === "token") pacer.append(evt.content);
            else if (evt.type === "error") turn.showError(evt.message);
            else if (evt.type === "done") pacer.finish();
          }
        }
      }
    } catch (err) {
      aborted = controller.signal.aborted;
      if (pacer) pacer.flush();
      if (!aborted) {
        turn.showError(
          err instanceof TypeError
            ? "Cannot reach the server. Is `uv run server.py` running?"
            : `Request failed: ${err.message}`,
        );
      }
    } finally {
      if (aborted) turn.showStopped();
      // The stream's full text is authoritative for storage; the DOM may
      // still be revealing the tail, and finalizes when the pacer drains.
      if (thisPacer.timer !== null) {
        persistTurn(turn, thread, thisPacer.buf);
      } else {
        finalizeTurn(turn, thread, thisPacer.buf);
      }
      streaming = false;
      controller = null;
      setStreamingUI(false);
      els.input.focus();
    }
  }

  function persistTurn(turn, thread, fullText) {
    if (turn.persisted) return;
    const hadContent = fullText.trim().length > 0 || turn.stepsMeta.length > 0;
    if (!hadContent) return;
    turn.persisted = true;
    thread.messages.push({
      role: "assistant",
      content: fullText,
      steps: turn.stepsMeta.map(({ label, detail }) => ({ label, detail })),
    });
    thread.updatedAt = Date.now();
    saveThreads();
    renderSidebar();
  }

  function finalizeTurn(turn, thread, fullText) {
    turn.finalize(fullText);
    persistTurn(turn, thread, fullText);
  }

  /* ---------- copy actions ---------- */

  els.messages.addEventListener("click", (e) => {
    const cc = e.target.closest(".copy-code");
    if (cc) {
      const code = cc.closest(".codeblock").querySelector("code");
      navigator.clipboard.writeText(code.textContent);
      cc.textContent = "Copied";
      setTimeout(() => (cc.textContent = "Copy"), 1200);
      return;
    }
    const ca = e.target.closest(".copy-all");
    if (ca) {
      const text = turnText.get(ca.closest(".turn-assistant"));
      if (text != null) {
        navigator.clipboard.writeText(text);
        ca.textContent = "Copied";
        setTimeout(() => (ca.textContent = "Copy"), 1200);
      }
    }
  });

  /* ---------- composer ---------- */

  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
  }

  els.input.addEventListener("input", () => {
    autosize();
    updateSendBtn();
  });

  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!streaming) send(els.input.value);
    }
  });

  els.sendBtn.addEventListener("click", () => {
    if (streaming) {
      if (controller) controller.abort();
    } else {
      send(els.input.value);
    }
  });

  els.newChatBtn.addEventListener("click", newChat);
  els.menuBtn.addEventListener("click", () => {
    els.sidebar.classList.add("open");
    els.scrim.classList.add("show");
  });
  els.scrim.addEventListener("click", closeMobileSidebar);

  function closeMobileSidebar() {
    els.sidebar.classList.remove("open");
    els.scrim.classList.remove("show");
  }

  /* ---------- init ---------- */

  currentId = localStorage.getItem(CURRENT_KEY) || null;
  if (currentId && !getThread(currentId)) currentId = null;
  if (!currentId && threads.length) {
    const latest = [...threads].sort((a, b) => b.updatedAt - a.updatedAt)[0];
    setCurrent(latest.id);
  }
  renderSidebar();
  renderThread(getThread(currentId));
  updateSendBtn();
  els.input.focus();
})();
