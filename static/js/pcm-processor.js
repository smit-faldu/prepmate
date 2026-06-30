/**
 * AudioWorklet processor for real-time Float32 → Int16 PCM conversion.
 * Replaces the deprecated ScriptProcessorNode.
 * Runs on the high-priority audio rendering thread.
 */
class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // Accumulate samples until we have a full 4096-sample chunk (~256ms @16kHz)
        this._buffer = [];
        this._chunkSize = 4096;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || input.length === 0) return true;

        const channelData = input[0]; // Mono channel
        for (let i = 0; i < channelData.length; i++) {
            this._buffer.push(channelData[i]);
        }

        // Send a chunk once we've accumulated enough samples
        while (this._buffer.length >= this._chunkSize) {
            const chunk = this._buffer.splice(0, this._chunkSize);
            const int16 = new Int16Array(chunk.length);
            for (let i = 0; i < chunk.length; i++) {
                const s = Math.max(-1, Math.min(1, chunk[i]));
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            // Transfer the buffer to avoid copying
            this.port.postMessage(int16.buffer, [int16.buffer]);
        }

        return true; // Keep processor alive
    }
}

registerProcessor('pcm-processor', PCMProcessor);
