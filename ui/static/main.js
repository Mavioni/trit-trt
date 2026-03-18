/* ── TRIT-TRT Frontend ─────────────────────────────────── */
(function () {
    "use strict";

    // ── DOM refs ──────────────────────────────────────────
    var chat = document.getElementById("chat");
    var promptInput = document.getElementById("prompt-input");
    var sendBtn = document.getElementById("send-btn");
    var settingsBtn = document.getElementById("settings-btn");
    var settingsPanel = document.getElementById("settings-panel");
    var settingsBackdrop = document.getElementById("settings-backdrop");
    var settingsClose = document.getElementById("settings-close");
    var knowledgeBtn = document.getElementById("knowledge-btn");
    var knowledgePanel = document.getElementById("knowledge-panel");
    var knowledgeBackdrop = document.getElementById("knowledge-backdrop");
    var knowledgeClose = document.getElementById("knowledge-close");
    var knowledgeBadge = document.getElementById("knowledge-badge");
    var knowledgeList = document.getElementById("knowledge-list");
    var connectionStatus = document.getElementById("connection-status");

    // ── State ─────────────────────────────────────────────
    var ws = null;
    var generating = false;
    var currentResponseEl = null;
    var roundCards = {};
    var knowledgeCount = 0;

    // ── WebSocket ─────────────────────────────────────────
    function connect() {
        var protocol = location.protocol === "https:" ? "wss:" : "ws:";
        var url = protocol + "//" + location.host + "/ws";
        ws = new WebSocket(url);

        ws.onopen = function () {
            connectionStatus.textContent = "connected";
            connectionStatus.className = "connection-status connected";
        };

        ws.onclose = function () {
            connectionStatus.textContent = "disconnected";
            connectionStatus.className = "connection-status disconnected";
            ws = null;
            if (generating) {
                resetInput();
                addSystemMessage("Connection lost.");
            }
        };

        ws.onerror = function () {
            connectionStatus.textContent = "disconnected";
            connectionStatus.className = "connection-status disconnected";
        };

        ws.onmessage = function (event) {
            var data;
            try {
                data = JSON.parse(event.data);
            } catch (e) {
                return;
            }
            handleEvent(data);
        };
    }

    // ── Event Dispatch ────────────────────────────────────
    function handleEvent(ev) {
        switch (ev.type) {
            case "status":    onStatus(ev); break;
            case "candidates": onCandidates(ev); break;
            case "selected":  onSelected(ev); break;
            case "insight":   onInsight(ev); break;
            case "result":    onResult(ev); break;
            case "error":     onError(ev); break;
            case "cancelled": onCancelled(); break;
        }
    }

    // ── Event: status ─────────────────────────────────────
    function onStatus(ev) {
        if (!currentResponseEl) {
            currentResponseEl = createResponseCard();
        }
        var roundEl = ensureRoundCard(ev.round, ev.total_rounds);
        updateRoundPhase(roundEl, ev.phase);
        autoScroll();
    }

    // ── Event: candidates ─────────────────────────────────
    function onCandidates(ev) {
        var roundEl = ensureRoundCard(ev.round, null);
        var body = roundEl.querySelector(".round-body");

        var toggle = document.createElement("div");
        toggle.className = "candidates-toggle";
        toggle.textContent = ev.count + " candidates generated (click to show)";

        var list = document.createElement("div");
        list.className = "candidates-list";

        (ev.texts || []).forEach(function (text, i) {
            var item = document.createElement("div");
            item.className = "candidate-item";
            item.textContent = "#" + (i + 1) + ": " + text;
            list.appendChild(item);
        });

        toggle.addEventListener("click", function () {
            var expanded = list.classList.toggle("expanded");
            toggle.textContent = ev.count + " candidates generated (click to " + (expanded ? "hide" : "show") + ")";
        });

        body.appendChild(toggle);
        body.appendChild(list);
        autoScroll();
    }

    // ── Event: selected ───────────────────────────────────
    function onSelected(ev) {
        var roundEl = ensureRoundCard(ev.round, null);
        var body = roundEl.querySelector(".round-body");

        var selDiv = document.createElement("div");
        selDiv.className = "selected-text";
        selDiv.textContent = ev.text;
        body.appendChild(selDiv);

        var confEl = createConfidenceBar(ev.confidence);
        body.appendChild(confEl);

        updateRoundPhase(roundEl, "selected");
        autoScroll();
    }

    // ── Event: insight ────────────────────────────────────
    function onInsight(ev) {
        var roundEl = ensureRoundCard(ev.round, null);
        var body = roundEl.querySelector(".round-body");
        var ins = document.createElement("div");
        ins.className = "insight-item";
        ins.textContent = ev.text;
        body.appendChild(ins);

        addKnowledgeItem(ev.text, ev.confidence);
        autoScroll();
    }

    // ── Event: result ─────────────────────────────────────
    function onResult(ev) {
        if (!currentResponseEl) {
            currentResponseEl = createResponseCard();
        }
        showFinalAnswer(currentResponseEl, ev.text, {
            confidence: ev.confidence,
            rounds_used: ev.rounds_used,
            total_candidates: ev.total_candidates,
            early_stopped: ev.early_stopped,
            knowledge_count: ev.knowledge_count,
        });
        resetInput();
        autoScroll();
    }

    // ── Event: error ──────────────────────────────────────
    function onError(ev) {
        var msg = document.createElement("div");
        msg.className = "message error";
        msg.textContent = "Error: " + (ev.message || "Unknown error");
        chat.appendChild(msg);
        resetInput();
        autoScroll();
    }

    // ── Event: cancelled ──────────────────────────────────
    function onCancelled() {
        addSystemMessage("Generation cancelled.");
        resetInput();
        autoScroll();
    }

    // ── Settings ──────────────────────────────────────────
    function getSettings() {
        return {
            rounds: parseInt(document.getElementById("setting-rounds").value, 10),
            candidates: parseInt(document.getElementById("setting-candidates").value, 10),
            max_tokens: parseInt(document.getElementById("setting-max-tokens").value, 10),
            temperature: parseFloat(document.getElementById("setting-temperature").value),
            selection_method: document.getElementById("setting-selection-method").value,
            reflection_depth: document.getElementById("setting-reflection-depth").value,
            early_stop_threshold: parseFloat(document.getElementById("setting-early-stop").value),
            knowledge_persistence: document.getElementById("setting-knowledge-persistence").checked,
        };
    }

    function initSliderSync() {
        var sliders = document.querySelectorAll('.setting input[type="range"]');
        sliders.forEach(function (slider) {
            var display = document.querySelector('.setting-value[data-for="' + slider.id + '"]');
            if (display) {
                slider.addEventListener("input", function () {
                    display.textContent = slider.value;
                });
            }
        });
    }

    // ── Send / Cancel ─────────────────────────────────────
    function send() {
        var text = promptInput.value.trim();
        if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

        addUserMessage(text);
        promptInput.value = "";
        promptInput.style.height = "auto";

        currentResponseEl = null;
        roundCards = {};

        var payload = {
            type: "generate",
            prompt: text,
            settings: getSettings(),
        };
        ws.send(JSON.stringify(payload));

        generating = true;
        sendBtn.textContent = "Cancel";
        sendBtn.classList.add("cancel");
        promptInput.disabled = true;
    }

    function cancel() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "cancel" }));
        }
    }

    function resetInput() {
        generating = false;
        sendBtn.textContent = "Send";
        sendBtn.classList.remove("cancel");
        promptInput.disabled = false;
        promptInput.focus();
    }

    // ── DOM Builders ──────────────────────────────────────
    function addUserMessage(text) {
        var msg = document.createElement("div");
        msg.className = "message user";
        msg.textContent = text;
        chat.appendChild(msg);
        autoScroll();
    }

    function addSystemMessage(text) {
        var msg = document.createElement("div");
        msg.className = "message system";
        msg.textContent = text;
        chat.appendChild(msg);
        autoScroll();
    }

    function createResponseCard() {
        var el = document.createElement("div");
        el.className = "message bot";
        chat.appendChild(el);
        return el;
    }

    function ensureRoundCard(round, totalRounds) {
        if (roundCards[round]) return roundCards[round];
        if (!currentResponseEl) {
            currentResponseEl = createResponseCard();
        }
        var card = addRoundCard(currentResponseEl, round, totalRounds);
        roundCards[round] = card;
        return card;
    }

    function addRoundCard(responseEl, round, totalRounds) {
        var card = document.createElement("div");
        card.className = "round-card";

        var header = document.createElement("div");
        header.className = "round-header";

        var title = document.createElement("span");
        title.className = "round-title";
        title.textContent = "Round " + round + (totalRounds ? " / " + totalRounds : "");

        var phase = document.createElement("span");
        phase.className = "round-phase";

        var spinner = document.createElement("span");
        spinner.className = "spinner";
        phase.appendChild(spinner);
        phase.appendChild(document.createTextNode("starting"));

        header.appendChild(title);
        header.appendChild(phase);

        var body = document.createElement("div");
        body.className = "round-body";

        header.addEventListener("click", function () {
            body.classList.toggle("collapsed");
        });

        card.appendChild(header);
        card.appendChild(body);
        responseEl.appendChild(card);
        return card;
    }

    function updateRoundPhase(roundEl, phaseText) {
        var phaseEl = roundEl.querySelector(".round-phase");
        if (!phaseEl) return;
        // Clear existing content
        while (phaseEl.firstChild) {
            phaseEl.removeChild(phaseEl.firstChild);
        }
        var isTerminal = phaseText === "selected" || phaseText === "done";
        if (!isTerminal) {
            var spinner = document.createElement("span");
            spinner.className = "spinner";
            phaseEl.appendChild(spinner);
        }
        phaseEl.appendChild(document.createTextNode(phaseText));
    }

    function createConfidenceBar(confidence) {
        var container = document.createElement("div");
        container.className = "confidence-bar-container";

        var track = document.createElement("div");
        track.className = "confidence-bar-track";

        var fill = document.createElement("div");
        fill.className = "confidence-bar-fill";
        setConfidenceBar(fill, confidence);

        track.appendChild(fill);

        var label = document.createElement("span");
        label.className = "confidence-label";
        label.textContent = Math.round(confidence * 100) + "%";
        label.style.color = confidenceColor(confidence);

        container.appendChild(track);
        container.appendChild(label);
        return container;
    }

    function setConfidenceBar(fillEl, confidence) {
        fillEl.style.width = Math.round(confidence * 100) + "%";
        fillEl.style.backgroundColor = confidenceColor(confidence);
    }

    function confidenceColor(c) {
        if (c < 0.5) return "var(--confidence-low)";
        if (c < 0.8) return "var(--confidence-mid)";
        return "var(--confidence-high)";
    }

    function showFinalAnswer(responseEl, text, meta) {
        var answer = document.createElement("pre");
        answer.className = "final-answer";
        answer.textContent = text;
        responseEl.appendChild(answer);

        var confBar = createConfidenceBar(meta.confidence);
        responseEl.appendChild(confBar);

        var metaBar = document.createElement("div");
        metaBar.className = "meta-bar";

        var items = [
            ["Confidence", Math.round(meta.confidence * 100) + "%"],
            ["Rounds", String(meta.rounds_used)],
            ["Candidates", String(meta.total_candidates)],
        ];
        if (meta.early_stopped) {
            items.push(["Early Stop", "yes"]);
        }
        if (meta.knowledge_count !== undefined) {
            items.push(["Knowledge", String(meta.knowledge_count)]);
        }

        items.forEach(function (pair) {
            var item = document.createElement("span");
            item.className = "meta-item";

            var labelSpan = document.createElement("span");
            labelSpan.className = "meta-label";
            labelSpan.textContent = pair[0] + ":";

            item.appendChild(labelSpan);
            item.appendChild(document.createTextNode(" " + pair[1]));
            metaBar.appendChild(item);
        });

        responseEl.appendChild(metaBar);
    }

    // ── Knowledge Panel ───────────────────────────────────
    function addKnowledgeItem(text, confidence) {
        knowledgeCount++;
        knowledgeBadge.textContent = String(knowledgeCount);

        var item = document.createElement("div");
        item.className = "knowledge-item";
        item.textContent = text;

        var confLabel = document.createElement("div");
        confLabel.className = "knowledge-confidence";
        confLabel.textContent = "Confidence: " + Math.round((confidence || 0) * 100) + "%";
        item.appendChild(confLabel);

        knowledgeList.appendChild(item);
    }

    // ── Auto-scroll ───────────────────────────────────────
    function autoScroll() {
        requestAnimationFrame(function () {
            chat.scrollTop = chat.scrollHeight;
        });
    }

    // ── Textarea Auto-resize ──────────────────────────────
    function autoResize() {
        promptInput.style.height = "auto";
        promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + "px";
    }

    // ── Panel Toggles ─────────────────────────────────────
    function toggleSettings() {
        settingsPanel.classList.toggle("hidden");
        settingsBackdrop.classList.toggle("hidden");
    }

    function closeSettings() {
        settingsPanel.classList.add("hidden");
        settingsBackdrop.classList.add("hidden");
    }

    function toggleKnowledge() {
        knowledgePanel.classList.toggle("hidden");
        knowledgeBackdrop.classList.toggle("hidden");
    }

    function closeKnowledge() {
        knowledgePanel.classList.add("hidden");
        knowledgeBackdrop.classList.add("hidden");
    }

    // ── Event Listeners ───────────────────────────────────
    sendBtn.addEventListener("click", function () {
        if (generating) {
            cancel();
        } else {
            send();
        }
    });

    promptInput.addEventListener("input", autoResize);

    promptInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (generating) return;
            send();
        }
    });

    settingsBtn.addEventListener("click", toggleSettings);
    settingsClose.addEventListener("click", closeSettings);
    settingsBackdrop.addEventListener("click", closeSettings);

    knowledgeBtn.addEventListener("click", toggleKnowledge);
    knowledgeClose.addEventListener("click", closeKnowledge);
    knowledgeBackdrop.addEventListener("click", closeKnowledge);

    // ── Init ──────────────────────────────────────────────
    initSliderSync();
    connect();
})();
