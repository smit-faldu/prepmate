/**
 * vision_client.js — PrepMate Webcam → MediaPipe Vision Client (v2)
 * ==================================================================
 * Updated for blendshape-based CV pipeline:
 *   - 8-expression vocabulary (smiling, excited, neutral, nervous,
 *     thinking, surprised, sad, angry)
 *   - Raw expression score breakdown mini-chart
 *   - VC mood badge sync (Marcus avatar reacts to detected emotion)
 *   - Blendshape quality indicator badge
 *
 * Opens a dedicated /ws/vision WebSocket alongside /ws/vc.
 * Captures webcam via getUserMedia → encodes frames as JPEG via canvas
 * → sends binary blobs at ~15fps → renders real-time expression/pose HUD.
 *
 * Usage:
 *   VisionClient.start(sessionId)   — call after session_id is known
 *   VisionClient.stop()             — call on session end / disconnect
 */

const VisionClient = (() => {
  // ─── State ────────────────────────────────────────────────────────────────
  let _ws          = null;
  let _stream      = null;
  let _videoEl     = null;
  let _canvasEl    = null;
  let _ctx         = null;
  let _captureLoop = null;
  let _sessionId   = null;
  let _running     = false;

  // Config
  const FPS              = 15;        // capture rate (browser → server)
  const JPEG_QUALITY     = 0.7;       // 0.7 balances quality vs bandwidth
  const FRAME_INTERVAL   = 1000 / FPS;
  const WS_RECONNECT_MS  = 3000;

  // ── Expression map: all 8 from blendshape pipeline ────────────────────────
  // Must stay in sync with _EXPRESSION_LABELS in mediapipe_vision_processor.py
  const EXPR_MAP = {
    smiling:   { emoji: "😊", label: "Smiling",    color: "#4ade80" },
    excited:   { emoji: "🤩", label: "Excited",    color: "#f59e0b" },
    neutral:   { emoji: "😐", label: "Neutral",    color: "#94a3b8" },
    nervous:   { emoji: "😰", label: "Nervous",    color: "#f87171" },
    thinking:  { emoji: "🤔", label: "Thinking",   color: "#a78bfa" },
    surprised: { emoji: "😲", label: "Surprised",  color: "#38bdf8" },
    sad:       { emoji: "😔", label: "Sad",         color: "#818cf8" },
    angry:     { emoji: "😤", label: "Frustrated",  color: "#fb7185" },
    unknown:   { emoji: "❓", label: "Unknown",    color: "#64748b" },
  };

  const POSE_MAP = {
    upright:         { emoji: "🧍", label: "Upright",          color: "#4ade80" },
    leaning_forward: { emoji: "🫱", label: "Leaning Forward",  color: "#38bdf8" },
    slouched:        { emoji: "🪑", label: "Slouching",        color: "#f87171" },
    gesturing:       { emoji: "👐", label: "Gesturing",        color: "#f59e0b" },
    tilted:          { emoji: "↗️", label: "Tilted",            color: "#a78bfa" },
    unknown:         { emoji: "❓", label: "Unknown",           color: "#64748b" },
  };

  // VC mood badge — expression → emoji shown on Marcus Reid's avatar
  const VC_MOOD_MAP = {
    smiling:   "😊",
    excited:   "🤩",
    neutral:   "🧐",
    nervous:   "🤨",
    thinking:  "🤔",
    surprised: "😲",
    sad:       "😕",
    angry:     "😒",
    unknown:   "🧐",
  };

  // ─── DOM helpers ──────────────────────────────────────────────────────────
  function _getEl(id) { return document.getElementById(id); }

  // ─── Raw score breakdown chart ────────────────────────────────────────────
  // Renders a mini horizontal bar chart from raw_expression_scores dict.
  function _renderScoreBars(rawScores) {
    const container = _getEl("vision-scores-chart");
    if (!container || !rawScores || typeof rawScores !== "object") return;

    // Sort by score descending, show top 6
    const entries = Object.entries(rawScores)
      .filter(([, v]) => v > 0.02)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 6);

    if (entries.length === 0) {
      container.innerHTML = `<p class="vscore-empty">Waiting for face…</p>`;
      return;
    }

    container.innerHTML = entries.map(([expr, score]) => {
      const info  = EXPR_MAP[expr] || { color: "#64748b", label: expr };
      const pct   = Math.round(score * 100);
      const label = info.label || (expr.charAt(0).toUpperCase() + expr.slice(1));
      return `
        <div class="vscore-row">
          <span class="vscore-label">${label}</span>
          <div class="vscore-track">
            <div class="vscore-fill" style="width:${pct}%;background:${info.color}88;box-shadow:0 0 6px ${info.color}44"></div>
          </div>
          <span class="vscore-pct" style="color:${info.color}">${pct}%</span>
        </div>`;
    }).join("");
  }

  // ─── Main HUD Updater ─────────────────────────────────────────────────────
  function _updateHUD(data) {
    const exprInfo = EXPR_MAP[data.expression] || EXPR_MAP.unknown;
    const poseInfo = POSE_MAP[data.pose]       || POSE_MAP.unknown;

    // ── Expression pill ──────────────────────────────────────────────────────
    const exprPill  = _getEl("vision-expr-pill");
    const exprEmoji = _getEl("vision-expr-emoji");
    const exprLabel = _getEl("vision-expr-label");
    const exprConf  = _getEl("vision-expr-conf");
    const exprBar   = _getEl("vision-expr-bar");

    if (exprPill) {
      exprPill.style.borderColor = exprInfo.color + "55";
      exprPill.style.boxShadow   = `0 0 12px ${exprInfo.color}1a`;
    }
    if (exprEmoji) exprEmoji.textContent = exprInfo.emoji;
    if (exprLabel) {
      exprLabel.textContent = exprInfo.label;
      exprLabel.style.color = exprInfo.color;
    }
    if (exprConf)  exprConf.textContent = `${Math.round(data.expression_confidence * 100)}%`;
    if (exprBar) {
      exprBar.style.width      = `${Math.round(data.expression_confidence * 100)}%`;
      exprBar.style.background = exprInfo.color;
    }

    // ── Pose pill ────────────────────────────────────────────────────────────
    const posePill  = _getEl("vision-pose-pill");
    const poseEmoji = _getEl("vision-pose-emoji");
    const poseLabel = _getEl("vision-pose-label");
    const poseConf  = _getEl("vision-pose-conf");
    const poseBar   = _getEl("vision-pose-bar");

    if (posePill) {
      posePill.style.borderColor = poseInfo.color + "55";
      posePill.style.boxShadow   = `0 0 12px ${poseInfo.color}1a`;
    }
    if (poseEmoji) poseEmoji.textContent = poseInfo.emoji;
    if (poseLabel) {
      poseLabel.textContent = poseInfo.label;
      poseLabel.style.color = poseInfo.color;
    }
    if (poseConf)  poseConf.textContent = `${Math.round(data.pose_confidence * 100)}%`;
    if (poseBar) {
      poseBar.style.width      = `${Math.round(data.pose_confidence * 100)}%`;
      poseBar.style.background = poseInfo.color;
    }

    // ── Raw blendshape score breakdown chart ──────────────────────────────────
    if (data.raw_expression_scores) {
      _renderScoreBars(data.raw_expression_scores);
    }

    // ── VC mood badge sync (Marcus avatar emoji) ──────────────────────────────
    const moodBadge = _getEl("vc-mood");
    if (moodBadge) {
      const moodEmoji = VC_MOOD_MAP[data.expression] || "🧐";
      if (moodBadge.textContent !== moodEmoji) {
        moodBadge.textContent = moodEmoji;
        moodBadge.classList.remove("mood-pop");
        void moodBadge.offsetWidth; // force reflow
        moodBadge.classList.add("mood-pop");
      }
    }

    // ── Signal quality badge ──────────────────────────────────────────────────
    const qualityEl = _getEl("vision-quality-badge");
    if (qualityEl) {
      const conf = data.expression_confidence || 0;
      if (conf >= 0.6) {
        qualityEl.textContent = "⚡ Blendshapes";
        qualityEl.className   = "vision-quality-badge quality-high";
      } else if (conf >= 0.3) {
        qualityEl.textContent = "🔍 Tracking";
        qualityEl.className   = "vision-quality-badge quality-mid";
      } else {
        qualityEl.textContent = "⚠ Low Signal";
        qualityEl.className   = "vision-quality-badge quality-low";
      }
    }

    // ── Camera dot + status ───────────────────────────────────────────────────
    const camDot = _getEl("vision-cam-dot");
    if (camDot) camDot.classList.add("active");

    const visionStatus = _getEl("vision-status-text");
    if (visionStatus) visionStatus.textContent = "Vision Active";

    // ── Flash HUD on each update ──────────────────────────────────────────────
    const hudCard = _getEl("vision-hud");
    if (hudCard) {
      hudCard.classList.remove("hud-flash");
      void hudCard.offsetWidth; // force reflow
      hudCard.classList.add("hud-flash");
    }
  }

  function _setVisionOffline() {
    const camDot = _getEl("vision-cam-dot");
    if (camDot) camDot.classList.remove("active");
    const visionStatus = _getEl("vision-status-text");
    if (visionStatus) visionStatus.textContent = "Camera Off";

    const qualityEl = _getEl("vision-quality-badge");
    if (qualityEl) {
      qualityEl.textContent = "Offline";
      qualityEl.className   = "vision-quality-badge quality-low";
    }
  }

  // ─── WebSocket ────────────────────────────────────────────────────────────
  function _connectWS(sessionId) {
    if (_ws) {
      try { _ws.close(); } catch (_) {}
    }

    const wsUrl = `ws://${location.host}/ws/vision?session_id=${sessionId}`;
    _ws = new WebSocket(wsUrl);
    _ws.binaryType = "arraybuffer";

    _ws.onopen = () => {
      console.log("[Vision WS] Connected →", wsUrl);
      _startCapture();
    };

    _ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "vision") {
          _updateHUD(msg);
        }
      } catch (_) {}
    };

    _ws.onerror = (e) => {
      console.warn("[Vision WS] Error:", e);
    };

    _ws.onclose = () => {
      console.warn("[Vision WS] Closed. Reconnecting in", WS_RECONNECT_MS, "ms…");
      _stopCapture();
      if (_running) {
        setTimeout(() => _connectWS(_sessionId), WS_RECONNECT_MS);
      }
    };
  }

  // ─── Webcam Capture ───────────────────────────────────────────────────────
  async function _initCamera() {
    try {
      _stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
        audio: false,
      });

      // Hidden video element for frame capture
      if (!_videoEl) {
        _videoEl = document.createElement("video");
        _videoEl.autoplay  = true;
        _videoEl.muted     = true;
        _videoEl.playsInline = true;
        _videoEl.style.display = "none";
        document.body.appendChild(_videoEl);
      }
      _videoEl.srcObject = _stream;
      await _videoEl.play();

      // Offscreen canvas for JPEG encoding
      if (!_canvasEl) {
        _canvasEl = document.createElement("canvas");
        _canvasEl.width  = 320;
        _canvasEl.height = 240;
      }
      _ctx = _canvasEl.getContext("2d");

      // Mirror webcam preview (if HUD preview element exists)
      const preview = _getEl("vision-cam-preview");
      if (preview) {
        preview.srcObject = _stream;
        preview.play().catch(() => {});
      }

      return true;
    } catch (err) {
      console.warn("[Vision] Camera access failed:", err.message);
      _setVisionOffline();
      return false;
    }
  }

  function _captureFrame() {
    if (!_videoEl || !_ctx || !_ws || _ws.readyState !== WebSocket.OPEN) return;
    if (_videoEl.videoWidth === 0) return;

    _ctx.drawImage(_videoEl, 0, 0, 320, 240);
    _canvasEl.toBlob(
      (blob) => {
        if (blob && _ws && _ws.readyState === WebSocket.OPEN) {
          _ws.send(blob);
        }
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  }

  function _startCapture() {
    if (_captureLoop) return;
    _captureLoop = setInterval(_captureFrame, FRAME_INTERVAL);
    console.log("[Vision] Capture loop started @", FPS, "fps");
  }

  function _stopCapture() {
    if (_captureLoop) {
      clearInterval(_captureLoop);
      _captureLoop = null;
    }
  }

  // ─── Public API ───────────────────────────────────────────────────────────
  async function start(sessionId) {
    if (_running) return;
    _running   = true;
    _sessionId = sessionId;

    console.log("[Vision] Starting vision pipeline for session:", sessionId);

    const camOk = await _initCamera();
    if (!camOk) {
      console.warn("[Vision] Running without camera — AI will use audio-only mode.");
      _running = false;
      return;
    }

    _connectWS(sessionId);
  }

  function stop() {
    _running = false;
    _stopCapture();
    _setVisionOffline();

    if (_ws) {
      try { _ws.close(); } catch (_) {}
      _ws = null;
    }
    if (_stream) {
      _stream.getTracks().forEach(t => t.stop());
      _stream = null;
    }
    if (_videoEl) {
      _videoEl.srcObject = null;
    }

    console.log("[Vision] Stopped.");
  }

  return { start, stop };
})();
