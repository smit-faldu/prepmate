// State variables
let ws = null;
let audioContext = null;
let mediaStream = null;
let processor = null;
let isRecording = false;

// DOM Elements
const recordBtn = document.getElementById('record-btn');
const btnText = document.getElementById('btn-text');
const clearBtn = document.getElementById('clear-btn');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const transcriptDisplay = document.getElementById('transcript-display');
const transcriptList = document.getElementById('transcript-list');
const emptyState = document.getElementById('empty-state');
const interimContainer = document.getElementById('interim-container');
const interimText = document.getElementById('interim-text');
const canvas = document.getElementById('visualizer');
const canvasCtx = canvas.getContext('2d');
const streamingLabel = document.getElementById('streaming-label');
const streamPulse = document.getElementById('stream-pulse');
const whisperModelName = document.getElementById('whisper-model-name');

// Sound visualization variables
let audioAnalyser = null;
let visualizerDataArray = null;
let animationFrameId = null;

// Initialize Visualizer in idle state
initVisualizer();

// Add Event Listeners
recordBtn.addEventListener('click', toggleRecording);
clearBtn.addEventListener('click', clearTranscripts);

// Fetch server-side Whisper model name to show in UI
fetch('/api/stt-info').then(r => r.json()).then(d => {
    if (whisperModelName && d.model) whisperModelName.textContent = d.model;
}).catch(() => {
    if (whisperModelName) whisperModelName.textContent = 'unknown';
});

// Sound Visualizer Setup (Drawing loop)
function initVisualizer() {
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    const draw = () => {
        animationFrameId = requestAnimationFrame(draw);

        const width = canvas.width;
        const height = canvas.height;

        canvasCtx.fillStyle = '#0b0b14'; // Dark background
        canvasCtx.fillRect(0, 0, width, height);

        canvasCtx.lineWidth = 2;

        if (isRecording && audioAnalyser) {
            // Active audio wave
            const bufferLength = audioAnalyser.frequencyBinCount;
            visualizerDataArray = new Uint8Array(bufferLength);
            audioAnalyser.getByteTimeDomainData(visualizerDataArray);

            // Draw a neat modern neon line wave
            canvasCtx.strokeStyle = '#00f0ff'; // Cyan accent
            canvasCtx.beginPath();

            const sliceWidth = width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = visualizerDataArray[i] / 128.0;
                const y = (v * height) / 2;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }

                x += sliceWidth;
            }

            canvasCtx.lineTo(width, height / 2);
            canvasCtx.stroke();

        } else {
            // Idle floating sine wave (calm pulse)
            canvasCtx.strokeStyle = '#7000ff'; // Purple accent (semi-translucent)
            canvasCtx.beginPath();

            const time = Date.now() * 0.004;
            const bufferLength = 100;
            const sliceWidth = width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const angle = (i / bufferLength) * Math.PI * 4 + time;
                // Fade out towards edges
                const edgeFactor = Math.sin((i / bufferLength) * Math.PI);
                const y = (height / 2) + Math.sin(angle) * 8 * edgeFactor;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }

                x += sliceWidth;
            }
            canvasCtx.stroke();
        }
    };

    draw();
}

// Toggle recording state
function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

// Start recording and WebSocket stream
async function startRecording() {
    try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        updateStatus('connecting', 'Connecting...');

        ws = new WebSocket(wsUrl);

        ws.onopen = async () => {
            try {
                // Get User Microphone
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 16000 // Request 16kHz
                    }
                });

                // Initialize Web Audio Context
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000 // Force sample rate to 16kHz
                });

                const source = audioContext.createMediaStreamSource(mediaStream);

                // Set up analyzer for visualization
                audioAnalyser = audioContext.createAnalyser();
                audioAnalyser.fftSize = 256;
                source.connect(audioAnalyser);

                // Load the AudioWorklet processor module (replaces deprecated ScriptProcessorNode)
                await audioContext.audioWorklet.addModule('/static/js/pcm-processor.js');

                // Create the worklet node: 1 input channel, 1 output channel
                processor = new AudioWorkletNode(audioContext, 'pcm-processor', {
                    numberOfInputs: 1,
                    numberOfOutputs: 1,
                    channelCount: 1,
                });
                source.connect(processor);
                // No need to connect processor to destination (we only capture, not playback)

                // Receive Int16 PCM chunks from the worklet and send over WebSocket
                processor.port.onmessage = (event) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(event.data); // event.data is an ArrayBuffer (Int16 PCM)
                    }
                };

                isRecording = true;
                recordBtn.classList.add('recording');
                btnText.textContent = 'Stop Listening';
                updateStatus('listening', 'Listening');
                emptyState.style.display = 'none';

            } catch (err) {
                console.error("Microphone access error:", err);
                alert("Microphone access denied or unavailable. Please enable permissions and try again.");
                stopRecording();
            }
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleServerMessage(data);
        };

        ws.onclose = () => {
            console.log("WebSocket connection closed");
            stopRecording();
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            stopRecording();
        };

    } catch (err) {
        console.error("Start recording failed:", err);
        stopRecording();
    }
}

// Stop recording and close connections
function stopRecording() {
    isRecording = false;
    recordBtn.classList.remove('recording');
    btnText.textContent = 'Start Listening';
    updateStatus('disconnected', 'Disconnected');

    // Stop mic stream tracks
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // Close audio context and processor
    if (processor) {
        processor.port.close();
        processor.disconnect();
        processor = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    audioAnalyser = null;

    // Close WebSocket
    if (ws) {
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
        }
        ws = null;
    }

    // Hide interim text panel
    interimContainer.style.display = 'none';
}

// Convert Float32Array browser audio samples to Int16Array PCM
function convertFloat32ToInt16(buffer) {
    const l = buffer.length;
    const buf = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        let s = Math.max(-1, Math.min(1, buffer[i]));
        buf[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return buf;
}

// Handle messages received from the Python pipeline server
function handleServerMessage(data) {
    if (data.type === 'interim') {
        const text = data.text.trim ? data.text.trim() : data.text;
        if (text) {
            interimContainer.style.display = 'flex';
            // Show text with blinking cursor to signal live streaming
            interimText.innerHTML = `${escapeHtml(text)}<span class="stream-cursor">&#9611;</span>`;
            // Update streaming badge
            if (streamingLabel) streamingLabel.textContent = 'Streaming…';
            if (streamPulse) streamPulse.classList.add('active');
        } else {
            interimContainer.style.display = 'none';
        }
    } else if (data.type === 'final') {
        const text = data.text.trim();
        if (text) {
            addTranscriptItem(text);
        }
        interimContainer.style.display = 'none';
        interimText.innerHTML = '';
        if (streamingLabel) streamingLabel.textContent = 'Processing…';
        if (streamPulse) streamPulse.classList.remove('active');
    } else if (data.type === 'status') {
        if (data.status === 'speaking') {
            updateStatus('listening', 'User Speaking...');
            if (streamingLabel) streamingLabel.textContent = 'Listening…';
        } else if (data.status === 'silence') {
            updateStatus('listening', 'Transcribing...');
            if (streamingLabel) streamingLabel.textContent = 'Idle';
            if (streamPulse) streamPulse.classList.remove('active');
        }
    }
}

// Escape HTML for safe text injection
function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

// Add a final transcription to the transcript display area
function addTranscriptItem(text) {
    emptyState.style.display = 'none';

    const timeString = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    const item = document.createElement('div');
    item.className = 'transcript-item';

    const meta = document.createElement('div');
    meta.className = 'transcript-meta';
    meta.textContent = timeString;

    const bubble = document.createElement('div');
    bubble.className = 'transcript-bubble';
    bubble.textContent = text;

    item.appendChild(meta);
    item.appendChild(bubble);

    transcriptList.appendChild(item);

    // Auto scroll to bottom
    transcriptDisplay.scrollTop = transcriptDisplay.scrollHeight;
}

// Clear all transcripts
function clearTranscripts() {
    transcriptList.innerHTML = '';
    emptyState.style.display = 'flex';
    interimContainer.style.display = 'none';
    interimText.textContent = '';
}

// Update the status UI indicator
function updateStatus(state, text) {
    statusIndicator.className = 'status-badge';

    if (state === 'connecting') {
        statusIndicator.classList.add('connected'); // Yellow/greenish
        statusText.textContent = text;
    } else if (state === 'listening') {
        statusIndicator.classList.add('listening'); // Red flashing
        statusText.textContent = text;
    } else if (state === 'disconnected') {
        statusIndicator.classList.add('disconnected'); // Red
        statusText.textContent = text;
    }
}
