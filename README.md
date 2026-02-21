# Shark Tank AI Voice Simulator

An interactive, real-time voice AI simulator that lets you pitch your startup idea to a virtual "Shark Tank" investor. The application leverages WebRTC for low-latency audio streaming and advanced Agentic AI architectures to maintain the structure of a professional 10-stage investment pitch.

## Features

- **Real-Time WebRTC Voice**: Uses [Pipecat](https://github.com/pipecat-ai/pipecat) over a WebRTC transport for ultra-fast, interruptible conversational audio.
- **Stage Progression Engine**: Enforces a strict 10-stage pitch framework (Introduction, Problem, Solution, Market, etc.).
- **Multi-Agent Architecture**: Built with [LangGraph](https://langchain-ai.github.io/langgraph/) using a prebuilt React Agent that dynamically evaluates your answers.
- **Tools Capability**: The Shark Agent actively decides when you've satisfied the current stage and triggers an `advance_pitch_stage` tool. If your pitch is terrible or you refuse to answer, it triggers a `drop_out` tool which forcefully severs the pipeline connection ("I'm out.").
- **Persistent Global Memory**: Built on top of `langgraph-checkpoint-sqlite`, the Virtual Shark remembers your pitch state, past answers, and conversational context across session disconnections.
- **Powered by Gemini**: Utilizes `gemini-3-flash-preview` via Google Generative AI for lightning fast, structured reasoning.
- **Local STT**: Runs Whisper locally (`tiny.en`) for immediate transcription and Voice Activity Detection (VAD).

## Project Structure

- `main.py`: FastAPI server entrypoint handling WebSocket signaling and REST routes.
- `bot/pipeline.py`: Configures and runs the Pipecat Task Pipeline orchestrating STT (Whisper), VAD (Silero), WebRTC transport, and the LangGraph processor.
- `bot/llm_processor.py`: Custom Pipecat `FrameProcessor`. Intercepts transcriptions, calls the LangGraph Agent via `ainvoke`, formats tool calls, updates state, and manages pipeline lifecycle events (like dropping out via `EndFrame`).
- `bot/agent.py`: LangChain / LangGraph definitions. Contains the core React Agent logic, Persona formatting (`SHARK_SYSTEM_PROMPT`), Tool definitions (`@tool`), and the SQLite Checkpointer attachment.
- `core/config.py`: Environment configuration management.

## Setup

1. **Install Requirements**: Ensure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have installed standard PyTorch for local Whisper STT execution).*

2. **Environment Variables**: Create a `.env` file in the root directory.
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   base_url=http://localhost:8000
   ```

3. **Run the Server**:
   ```bash
   python main.py
   # or
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Connect**: Open your browser, navigate to the frontend interface connected to your `base_url`, and start your pitch!

## The 10 Pitch Stages

1. Introduction
2. Problem
3. Solution
4. Market Size
5. Product/Demo
6. Business Model
7. Competition
8. Team
9. Financials/Metrics
10. The Ask/Roadmap

The Shark will organically pull you through these stages, but be prepared—if you are vague, it will grill you before advancing.
