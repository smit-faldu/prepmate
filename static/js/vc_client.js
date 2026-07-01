/**
 * vc_client.js — PrepMate VC Pitch Arena
 * Handles: WebSocket session, audio capture, message rendering,
 *          stage tracking, pitch intelligence UI, and "I'm out" animations.
 */

'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
let ws           = null;
let audioContext = null;
let mediaStream  = null;
let processor    = null;
let audioAnalyser= null;
let animFrameId  = null;
let isRecording  = false;
let sessionId    = null;
let currentStage = 'intro';
let renderedRedFlags = new Set();

// ─── TTS State ────────────────────────────────────────────────────────────────
let isMuted        = false;   // user-toggled mute
let ttsPlaying     = false;   // true while Marcus is speaking
let micGated       = false;   // true while mic is disabled waiting for TTS
let ttsPlayer      = null;    // TtsPlayer instance, created on first connect

// ─── DOM References ───────────────────────────────────────────────────────────
const micBtn          = document.getElementById('mic-btn');
const micIcon         = document.getElementById('mic-icon');
const controlsLabel   = document.getElementById('controls-label');
const controlsSubLabel= document.getElementById('controls-sublabel');
const chatFeed        = document.getElementById('chat-feed');
const chatEmpty       = document.getElementById('chat-empty');
const vcThinking      = document.getElementById('vc-thinking');
const interimContent  = document.getElementById('interim-content');
const liveIndicator   = document.getElementById('live-indicator');
const exchangeCount   = document.getElementById('exchange-count');
const connStatus      = document.getElementById('conn-status');
const connText        = document.getElementById('conn-text');
const connStatus2     = document.getElementById('conn-status-2');
const connText2       = document.getElementById('conn-text-2');
const vcStatusPill    = document.getElementById('vc-status-pill');
const vcStatusText    = document.getElementById('vc-status-text');
const vcMood          = document.getElementById('vc-mood');
const stageBadge      = document.getElementById('stage-badge');
const stageDesc       = document.getElementById('stage-desc');
const intelProduct    = document.getElementById('intel-product');
const intelProductVal = document.getElementById('intel-product-val');
const intelMarket     = document.getElementById('intel-market');
const intelMarketVal  = document.getElementById('intel-market-val');
const intelAsk        = document.getElementById('intel-ask');
const intelAskVal     = document.getElementById('intel-ask-val');
const intelTraction   = document.getElementById('intel-traction');
const intelTractionVal= document.getElementById('intel-traction-val');
const intelClarity    = document.getElementById('intel-clarity');
const clarityFill     = document.getElementById('clarity-fill');
const clarityLabel    = document.getElementById('clarity-score-label');
const intelFlags      = document.getElementById('intel-flags');
const redFlagsList    = document.getElementById('red-flags-list');
const marketBadge     = document.getElementById('market-badge');
const outOverlay      = document.getElementById('out-overlay');
const outReason       = document.getElementById('out-reason');
const canvas          = document.getElementById('vc-visualizer');
const canvasCtx       = canvas.getContext('2d');

// Stage step elements
const stageSteps = {
  intro:        document.getElementById('step-intro'),
  deep_dive:    document.getElementById('step-deep_dive'),
  negotiation:  document.getElementById('step-negotiation'),
  decision:     document.getElementById('step-decision'),
};

const STAGE_META = {
  intro: {
    icon: '🌱',
    label: 'Introduction',
    desc: 'Introduce your product, target audience, and funding ask clearly.',
    order: 0,
  },
  deep_dive: {
    icon: '🔬',
    label: 'Deep Dive',
    desc: 'Defend your market size, moat, and unit economics under pressure.',
    order: 1,
  },
  negotiation: {
    icon: '🤝',
    label: 'Negotiation',
    desc: 'Justify your valuation. Tell me exactly what you\'ll do with the money.',
    order: 2,
  },
  decision: {
    icon: '⚖️',
    label: 'Final Decision',
    desc: 'This is it — deal or no deal. Make your case one last time.',
    order: 3,
  },
};

// ─── TtsPlayer ────────────────────────────────────────────────────────────────
// Decodes incoming MP3 binary frames and plays them via Web Audio API.
// Turn-based: queues chunks, plays sequentially, signals done via onFinished().
// ──────────────────────────────────────────────────────────────────────────────
class TtsPlayer {
  constructor(ctx) {
    this._ctx        = ctx;       // shared AudioContext
    this._chunks     = [];        // incoming MP3 byte arrays
    this._mp3Buffer  = [];        // accumulating raw MP3 bytes
    this._source     = null;      // currently playing AudioBufferSourceNode
    this._playing    = false;
    this._done       = false;     // server sent tts_done
    this._gainNode   = ctx.createGain();
    this._gainNode.connect(ctx.destination);
  }

  /** Feed an MP3 chunk (ArrayBuffer) from the server */
  feedChunk(arrayBuffer) {
    const bytes = new Uint8Array(arrayBuffer);
    this._mp3Buffer.push(...bytes);
    if (!this._playing) this._tryFlush();
  }

  /** Called when the server sends tts_done — signals no more chunks. */
  markDone() {
    this._done = true;
    if (!this._playing) this._tryFlush();
  }

  /** Decode & play everything accumulated so far, then call _onPlaybackEnd. */
  _tryFlush() {
    if (this._mp3Buffer.length === 0) {
      if (this._done) this._onPlaybackEnd();
      return;
    }
    const allBytes = new Uint8Array(this._mp3Buffer);
    this._mp3Buffer = [];
    this._playing = true;

    this._ctx.decodeAudioData(
      allBytes.buffer.slice(0),
      (audioBuffer) => {
        if (isMuted) {
          // Muted: skip playback but honour timing so tts_done still fires
          const durationMs = Math.round(audioBuffer.duration * 1000);
          setTimeout(() => {
            this._playing = false;
            this._tryFlush();
          }, durationMs);
          return;
        }

        const source = this._ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this._gainNode);
        source.onended = () => {
          this._playing = false;
          this._source  = null;
          this._tryFlush();
        };
        this._source = source;
        source.start();
      },
      (err) => {
        console.warn('[TTS] decodeAudioData error, skipping chunk:', err);
        this._playing = false;
        this._tryFlush();
      }
    );
  }

  /** Invoked after the final chunk finishes playing. */
  _onPlaybackEnd() {
    ttsPlaying = false;
    micGated   = false;
    console.debug('[TTS] Playback complete — mic re-enabled');
    // Re-enable mic UI feedback
    if (isRecording) {
      controlsLabel.textContent    = 'Pitching...';
      controlsSubLabel.textContent = 'Speak naturally, pause to submit';
      updateStatus('connected', 'Your turn — speak now');
    }
  }

  /** Mute/unmute the gain without stopping playback. */
  setMuted(muted) {
    this._gainNode.gain.setTargetAtTime(muted ? 0 : 1, this._ctx.currentTime, 0.02);
  }

  /** Hard stop (e.g. session end). */
  stop() {
    if (this._source) {
      try { this._source.stop(); } catch (_) {}
      this._source = null;
    }
    this._mp3Buffer = [];
    this._chunks    = [];
    this._playing   = false;
    this._done      = false;
    ttsPlaying      = false;
    micGated        = false;
  }
}

// ─── Initialise Visualizer ───────────────────────────────────────────────────
drawIdleWave();

// ─── Event Listeners ─────────────────────────────────────────────────────────
micBtn.addEventListener('click', togglePitch);

// ─── Session Management ───────────────────────────────────────────────────────
async function createSession() {
  const res = await fetch('/api/vc/session', { method: 'POST' });
  const data = await res.json();
  return data.session_id;
}

// ─── Toggle Pitch ─────────────────────────────────────────────────────────────
async function togglePitch() {
  if (isRecording) {
    stopPitch();
  } else {
    await startPitch();
  }
}

async function startPitch() {
  try {
    updateStatus('connecting', 'Connecting...');

    // Create a new server-side session
    sessionId = await createSession();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/vc?session_id=${sessionId}`;

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';   // receive MP3 chunks as ArrayBuffer

    ws.onopen = async () => {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 16000,
          },
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: 16000,
        });

        // Shared AudioContext is also used by TtsPlayer
        ttsPlayer = new TtsPlayer(audioContext);
        ttsPlayer.setMuted(isMuted);

        const source = audioContext.createMediaStreamSource(mediaStream);

        // Analyser for visualizer
        audioAnalyser = audioContext.createAnalyser();
        audioAnalyser.fftSize = 256;
        source.connect(audioAnalyser);

        // AudioWorklet for PCM capture
        await audioContext.audioWorklet.addModule('/static/js/pcm-processor.js');
        processor = new AudioWorkletNode(audioContext, 'pcm-processor', {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
        });
        source.connect(processor);

        processor.port.onmessage = (ev) => {
          // Only send audio when mic is not gated (i.e., Marcus is not speaking)
          if (ws && ws.readyState === WebSocket.OPEN && !micGated) {
            ws.send(ev.data);
          }
        };

        // UI: recording state
        isRecording = true;
        micBtn.classList.add('recording');
        micBtn.setAttribute('aria-pressed', 'true');
        micBtn.setAttribute('aria-label', 'Stop recording');
        micIcon.textContent = '⏹';
        controlsLabel.textContent = 'Pitching...';
        controlsSubLabel.textContent = 'Speak naturally, pause to submit';
        liveIndicator.classList.add('active');
        chatEmpty.style.display = 'none';

        updateStatus('connected', 'Connected');
      } catch (err) {
        console.error('Mic error:', err);
        alert('Microphone access denied. Please enable mic permissions and try again.');
        stopPitch();
      }
    };

    ws.onmessage = (ev) => {
      // Binary frame = MP3 audio chunk from ElevenLabs TTS
      if (ev.data instanceof ArrayBuffer) {
        if (ttsPlayer) ttsPlayer.feedChunk(ev.data);
        return;
      }
      // Text frame = JSON control/data message
      try {
        const data = JSON.parse(ev.data);
        handleServerMessage(data);
      } catch (e) {
        console.warn('WS parse error:', e);
      }
    };

    ws.onclose  = () => stopPitch();
    ws.onerror  = (e) => { console.error('WS error:', e); stopPitch(); };

  } catch (err) {
    console.error('startPitch failed:', err);
    stopPitch();
  }
}


function stopPitch() {
  isRecording = false;
  micGated    = false;
  ttsPlaying  = false;
  micBtn.classList.remove('recording');
  micBtn.setAttribute('aria-pressed', 'false');
  micBtn.setAttribute('aria-label', 'Start recording microphone');
  micIcon.textContent = '🎤';
  controlsLabel.textContent = 'Start Pitch';
  controlsSubLabel.textContent = 'Click mic & speak clearly';
  liveIndicator.classList.remove('active');

  if (ttsPlayer) { ttsPlayer.stop(); ttsPlayer = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (processor)   { processor.port.close(); processor.disconnect(); processor = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  audioAnalyser = null;

  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    ws.close();
  }
  ws = null;

  interimContent.textContent  = 'Session ended.';
  interimContent.classList.remove('active');
  vcThinking.classList.remove('active');
  updateStatus('disconnected', 'Disconnected');
}


// ─── Server Message Dispatcher ────────────────────────────────────────────────
function handleServerMessage(data) {
  switch (data.type) {
    case 'interim':
      handleInterim(data.text);
      break;

    case 'status':
      handleVADStatus(data.status);
      break;

    case 'vc_thinking':
      handleThinking(data.founder_text);
      break;

    case 'vc_token':
      handleVCToken(data.text);
      break;

    case 'vc_response':
      handleVCResponse(data);
      break;

    case 'tts_done':
      // Server finished sending all TTS audio for this turn.
      // Let TtsPlayer handle the actual playback-end callback;
      // just mark the stream as complete so it can flush its buffer.
      if (ttsPlayer) ttsPlayer.markDone();
      else {
        // TTS was disabled server-side — re-enable mic immediately
        micGated   = false;
        ttsPlaying = false;
        if (isRecording) {
          controlsLabel.textContent    = 'Pitching...';
          controlsSubLabel.textContent = 'Speak naturally, pause to submit';
          updateStatus('connected', 'Your turn — speak now');
        }
      }
      break;

    case 'error':
      console.error('Server error:', data.message);
      appendSystemMessage(`⚠ Error: ${data.message}`);
      // On error, always re-enable mic so the session isn't stuck
      micGated   = false;
      ttsPlaying = false;
      break;
  }
}

function handleInterim(text) {
  if (text && text.trim()) {
    interimContent.textContent = `"${text}"`;
    interimContent.classList.add('active');
  }
}

function handleVADStatus(status) {
  if (status === 'speaking') {
    updateStatus('speaking', 'Listening...');
    vcThinking.classList.remove('active');
  } else if (status === 'silence') {
    updateStatus('connected', 'Processing...');
    interimContent.textContent = 'Processing your words...';
    interimContent.classList.remove('active');
  }
}

function handleThinking(founderText) {
  // Add founder bubble
  if (founderText) {
    appendFounderMessage(founderText);
    interimContent.textContent = '';
    interimContent.classList.remove('active');
  }
  vcThinking.classList.add('active');
  updateStatus('thinking', 'Marcus is thinking...');
  setVCMood('thinking');
  // Gate the mic — Marcus has the floor now (turn-based)
  micGated   = true;
  ttsPlaying = true;
  controlsLabel.textContent    = 'Marcus is speaking...';
  controlsSubLabel.textContent = 'Wait for Marcus to finish';
  // Reset streaming bubble ref — a fresh one is created on the first token
  _streamingVCBubble = null;
  _streamingVCText = '';
}


// ─── Streaming VC reply (token-by-token) ───────────────────────────────────────
// As soon as the first token arrives we create the message bubble and start
// filling it in live, instead of waiting for the full reply + analyst pass
// to finish. This is where the latency fix is actually felt by the user.
let _streamingVCBubble = null;
let _streamingVCText = '';

function handleVCToken(text) {
  if (!text) return;

  // First token: stop the "thinking" indicator and create the bubble now.
  if (!_streamingVCBubble) {
    vcThinking.classList.remove('active');
    updateStatus('connected', 'Marcus is responding...');
    chatEmpty.style.display = 'none';

    const group = document.createElement('div');
    group.className = 'message-group vc streaming';
    group.innerHTML = `
      <div class="message-sender">Marcus Reid</div>
      <div class="message-bubble"></div>
      <div class="message-time">${timeNow()}</div>
    `;
    chatFeed.appendChild(group);
    _streamingVCBubble = group.querySelector('.message-bubble');
    _streamingVCText = '';
  }

  _streamingVCText += text;
  // Strip END_PITCH tag live in case it streams in mid-token
  _streamingVCBubble.textContent = _streamingVCText.replace(/<END_PITCH>/g, '').trim();
  scrollChat();
}

function handleVCResponse(data) {
  vcThinking.classList.remove('active');

  // Update pitch intelligence panel
  if (data.pitch_metrics) {
    updateIntelPanel(data.pitch_metrics);
  }

  // Update stage
  if (data.stage && data.stage !== currentStage) {
    transitionStage(data.stage);
  }

  // Exchange count
  if (data.exchange_count !== undefined) {
    exchangeCount.textContent = `Turn ${data.exchange_count}`;
  }

  const vcText = data.vc_text || '';
  const isOut  = data.is_out || vcText.toLowerCase().includes("i'm out") || vcText.toLowerCase().includes("im out");

  // If tokens already streamed in, just finalize that bubble (mark it
  // not-streaming + correct any drift vs. the authoritative final text)
  // instead of appending a second duplicate message.
  if (_streamingVCBubble) {
    const group = _streamingVCBubble.closest('.message-group');
    if (group) group.classList.remove('streaming');
    if (isOut) group?.classList.add('is-out');
    _streamingVCBubble.textContent = vcText.replace(/<END_PITCH>/g, '').trim();
    _streamingVCBubble = null;
    _streamingVCText = '';
  } else {
    // Fallback: no tokens arrived (e.g. fallback path in vc_agent.py) —
    // render the full message as before.
    appendVCMessage(vcText, isOut);
  }

  if (data.latency_ms !== undefined) {
    console.debug(`[VC] Turn latency: ${data.latency_ms}ms`);
  }

  // Update VC mood + status
  setVCMood(isOut ? 'out' : (currentStage === 'deep_dive' ? 'skeptical' : 'interested'));

  updateStatus('connected', 'Listening...');

  // Trigger "I'm Out" overlay
  if (data.pitch_ended || isOut) {
    setTimeout(() => triggerOutOverlay(vcText), 800);
  }
}

// ─── Message Rendering ─────────────────────────────────────────────────────────
function appendFounderMessage(text) {
  chatEmpty.style.display = 'none';
  const group = document.createElement('div');
  group.className = 'message-group founder';
  group.innerHTML = `
    <div class="message-sender">You (Founder)</div>
    <div class="message-bubble">${escapeHtml(text)}</div>
    <div class="message-time">${timeNow()}</div>
  `;
  chatFeed.appendChild(group);
  scrollChat();
}

function appendVCMessage(text, isOut = false) {
  // Strip the <END_PITCH> tag from display
  const cleanText = text.replace(/<END_PITCH>/g, '').trim();
  const group = document.createElement('div');
  group.className = `message-group vc${isOut ? ' is-out' : ''}`;
  group.innerHTML = `
    <div class="message-sender">Marcus Reid</div>
    <div class="message-bubble">${escapeHtml(cleanText)}</div>
    <div class="message-time">${timeNow()}</div>
  `;
  chatFeed.appendChild(group);
  scrollChat();
}

function appendSystemMessage(text) {
  const el = document.createElement('div');
  el.style.cssText = 'text-align:center; font-size:12px; color:var(--text-dim); padding:8px 0;';
  el.textContent = text;
  chatFeed.appendChild(el);
  scrollChat();
}

function scrollChat() {
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

// ─── Stage Transition ─────────────────────────────────────────────────────────
function transitionStage(newStage) {
  const stageOrder = ['intro', 'deep_dive', 'negotiation', 'decision'];
  const oldIdx = stageOrder.indexOf(currentStage);
  const newIdx = stageOrder.indexOf(newStage);

  // Mark completed steps
  stageOrder.forEach((s, idx) => {
    const el = stageSteps[s];
    if (!el) return;
    el.classList.remove('active', 'completed');
    if (idx < newIdx)       el.classList.add('completed');
    else if (idx === newIdx) el.classList.add('active');
  });

  currentStage = newStage;

  // Update stage badge in right panel
  const meta = STAGE_META[newStage] || STAGE_META['intro'];
  stageBadge.setAttribute('data-stage', newStage);
  stageBadge.textContent = `${meta.icon} ${meta.label}`;
  stageDesc.textContent = meta.desc;

  // Add stage transition system message if advancing
  if (newIdx > oldIdx) {
    appendSystemMessage(`— Stage Advanced: ${meta.label} —`);
  }
}

// ─── Pitch Intelligence Panel ─────────────────────────────────────────────────
function updateIntelPanel(metrics) {
  function setField(el, valEl, value) {
    if (value && value !== 'not mentioned') {
      valEl.textContent = value;
      el.classList.add('visible');
    }
  }

  setField(intelProduct,  intelProductVal,  metrics.product);
  setField(intelMarket,   intelMarketVal,   metrics.market_size);
  setField(intelAsk,      intelAskVal,      metrics.ask);
  setField(intelTraction, intelTractionVal, metrics.traction);

  // Clarity score bar
  if (metrics.clarity_score) {
    const pct = (metrics.clarity_score / 10) * 100;
    clarityFill.style.width = `${pct}%`;
    clarityLabel.textContent = `${metrics.clarity_score}/10`;
    intelClarity.classList.add('visible');
  }

  // Red flags
  const flags = metrics.red_flags || [];
  if (flags.length > 0) {
    intelFlags.style.display = 'flex';
    intelFlags.classList.add('visible');
    flags.forEach(flag => {
      if (!renderedRedFlags.has(flag)) {
        renderedRedFlags.add(flag);
        const chip = document.createElement('div');
        chip.className = 'red-flag-chip';
        chip.textContent = flag;
        chip.style.animationDelay = `${(renderedRedFlags.size - 1) * 0.1}s`;
        redFlagsList.appendChild(chip);
      }
    });
  }

  // Market validity badge
  const mv = metrics.market_validity;
  if (mv && mv !== 'not_mentioned') {
    const labels = {
      credible:   '✓ Market Credible',
      inflated:   '↑ Market Inflated',
      unverified: '? Market Unverified',
    };
    marketBadge.textContent = labels[mv] || mv;
    marketBadge.className = `market-badge ${mv}`;
    marketBadge.style.display = 'block';
  }
}

// ─── VC Mood & Status ─────────────────────────────────────────────────────────
function setVCMood(mood) {
  const moods = {
    interested: { icon: '🧐', pill: 'interested', text: 'Interested' },
    skeptical:  { icon: '🤨', pill: 'skeptical',  text: 'Skeptical'  },
    thinking:   { icon: '🤔', pill: 'interested', text: 'Evaluating' },
    out:        { icon: '😤', pill: 'out',         text: 'Out'        },
  };
  const m = moods[mood] || moods.interested;
  vcMood.textContent = m.icon;
  vcStatusPill.className = `vc-status-pill ${m.pill}`;
  // Re-add dot
  vcStatusPill.innerHTML = `<span class="pill-dot"></span><span id="vc-status-text">${m.text}</span>`;
}

// ─── "I'm Out" Overlay ────────────────────────────────────────────────────────
function triggerOutOverlay(vcText) {
  // Extract the reason from the response text (everything before <END_PITCH>)
  let reason = vcText.replace(/<END_PITCH>/g, '').trim();
  // Remove "I'm out." prefix for the reason display
  reason = reason.replace(/^i['']m\s+out\.?\s*/i, '').trim();
  if (!reason) reason = 'The pitch session has ended. Review the feedback to improve your pitch.';

  outReason.textContent = reason;
  outOverlay.classList.add('active');
  outOverlay.setAttribute('aria-hidden', 'false');

  stopPitch();
  document.body.style.overflow = 'hidden';
}

function dismissOut() {
  outOverlay.classList.remove('active');
  outOverlay.setAttribute('aria-hidden', 'true');
  document.body.style.overflow = '';
  // Reset state for a fresh session
  resetUI();
}

window.dismissOut = dismissOut;

function resetUI() {
  chatFeed.innerHTML = '';
  chatEmpty.style.display = '';
  chatFeed.appendChild(chatEmpty);
  renderedRedFlags.clear();
  redFlagsList.innerHTML = '';
  intelFlags.style.display = 'none';
  intelProductVal.textContent  = '—';
  intelMarketVal.textContent   = '—';
  intelAskVal.textContent      = '—';
  intelTractionVal.textContent = '—';
  clarityFill.style.width = '0%';
  clarityLabel.textContent = '—';
  marketBadge.className = 'market-badge not_mentioned';
  marketBadge.style.display = 'none';
  exchangeCount.textContent = 'Turn 0';
  interimContent.textContent = 'Start speaking to pitch your idea...';
  [intelProduct, intelMarket, intelAsk, intelTraction, intelClarity, intelFlags].forEach(el => el.classList.remove('visible'));
  transitionStage('intro');
  setVCMood('interested');
}

// ─── Status / Connection Indicators ──────────────────────────────────────────
function updateStatus(state, label) {
  const statuses = ['connected', 'speaking', 'thinking'];
  [connStatus, connStatus2].forEach(el => {
    el.className = 'conn-status';
    if (statuses.includes(state)) el.classList.add(state);
  });
  connText.textContent  = label;
  connText2.textContent = label;
}

// ─── Audio Visualizer ─────────────────────────────────────────────────────────
function drawIdleWave() {
  if (animFrameId) cancelAnimationFrame(animFrameId);

  const w = canvas.width;
  const h = canvas.height;

  const draw = () => {
    animFrameId = requestAnimationFrame(draw);
    canvasCtx.clearRect(0, 0, w, h);
    canvasCtx.fillStyle = '#070710';
    canvasCtx.fillRect(0, 0, w, h);

    if (isRecording && audioAnalyser) {
      // Live waveform
      const buf = new Uint8Array(audioAnalyser.frequencyBinCount);
      audioAnalyser.getByteTimeDomainData(buf);

      canvasCtx.strokeStyle = '#c9a227';
      canvasCtx.lineWidth = 2;
      canvasCtx.shadowColor = '#c9a227';
      canvasCtx.shadowBlur = 8;
      canvasCtx.beginPath();

      const sw = w / buf.length;
      let x = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = buf[i] / 128.0;
        const y = (v * h) / 2;
        i === 0 ? canvasCtx.moveTo(x, y) : canvasCtx.lineTo(x, y);
        x += sw;
      }
      canvasCtx.lineTo(w, h / 2);
      canvasCtx.stroke();
      canvasCtx.shadowBlur = 0;

    } else {
      // Idle ambient wave (gold)
      const t = Date.now() * 0.002;
      canvasCtx.strokeStyle = 'rgba(201,162,39,0.35)';
      canvasCtx.lineWidth = 1.5;
      canvasCtx.beginPath();

      const steps = 120;
      const sw = w / steps;
      let x = 0;
      for (let i = 0; i < steps; i++) {
        const ef = Math.sin((i / steps) * Math.PI);
        const y = (h / 2) + Math.sin((i / steps) * Math.PI * 5 + t) * 6 * ef
                           + Math.sin((i / steps) * Math.PI * 11 + t * 1.3) * 3 * ef;
        i === 0 ? canvasCtx.moveTo(x, y) : canvasCtx.lineTo(x, y);
        x += sw;
      }
      canvasCtx.stroke();
    }
  };

  draw();
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function timeNow() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ─── Mute Toggle ─────────────────────────────────────────────────────────────
function toggleMute() {
  isMuted = !isMuted;
  const muteBtn  = document.getElementById('mute-btn');
  const muteIcon = document.getElementById('mute-icon');
  if (isMuted) {
    muteIcon.textContent = '🔇';
    muteBtn.classList.add('muted');
    muteBtn.setAttribute('aria-label', 'Unmute voice');
  } else {
    muteIcon.textContent = '🔊';
    muteBtn.classList.remove('muted');
    muteBtn.setAttribute('aria-label', 'Mute voice');
  }
  if (ttsPlayer) ttsPlayer.setMuted(isMuted);
  console.debug(`[TTS] ${isMuted ? 'Muted' : 'Unmuted'}`);
}
window.toggleMute = toggleMute;