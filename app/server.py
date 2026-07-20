"""
app/server.py — FastAPI application factory, static file mounts, and HTTP routes.

All WebSocket routes are registered here by importing their endpoint handlers
from the websockets sub-package. This file is the only place that knows about
FastAPI; the sub-packages are framework-agnostic where possible.
"""

import base64
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import resolve_whisper_config  # noqa: F401
from app.stt.streaming_processor import INTERIM_INTERVAL_SECS
from app.vc import new_session
from app.websockets.stt_ws import stt_websocket_endpoint
from app.websockets.vc_ws import vc_websocket_endpoint
from app.websockets.vision_ws import vision_websocket_endpoint

app = FastAPI(title="PrepMate — STT + VC Pitch Evaluator")

# ── Static assets ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Startup: warm the shared Whisper model ───────────────────────────────────
@app.on_event("startup")
async def _preload_stt_model() -> None:
    """
    Load the Whisper model once, before the first request arrives.

    Without this, the model loads lazily on the first websocket connection
    (whichever of /ws or /ws/vc gets hit first) and that user eats the full
    load time — several seconds, or minutes on a first-ever download. Every
    connection after that reuses the same cached model (see
    app/stt/streaming_processor.py: get_shared_whisper_model).
    """
    from app.stt.streaming_processor import preload_whisper_model

    wcfg = resolve_whisper_config()
    logger.info("[Startup] Pre-loading shared Whisper model …")
    await preload_whisper_model(wcfg["model"], wcfg["device"], wcfg["compute_type"])
    logger.info("[Startup] Whisper model ready.")


# ── Utility routes ────────────────────────────────────────────────────────────

@app.get("/favicon.ico")
async def favicon():
    """Minimal 1×1 transparent ICO — suppresses browser 404 noise."""
    ico_b64 = (
        "AAABAAEAAQEAAAEAGAAoAAAAFgAAACgAAAABAAAAAgAAAAEAGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    )
    return Response(content=base64.b64decode(ico_b64), media_type="image/x-icon")


# ── Page routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=RedirectResponse)
async def get_index():
    """Redirect root to the VC Pitch Arena (default landing page)."""
    return RedirectResponse(url="/vc")


@app.get("/vc", response_class=HTMLResponse)
async def get_vc():
    """Serve the VC Pitch Arena page."""
    path = os.path.join("templates", "vc.html")
    if not os.path.exists(path):
        return HTMLResponse(content="<h1>templates/vc.html not found!</h1>", status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── REST API routes ───────────────────────────────────────────────────────────

@app.get("/api/stt-info")
async def stt_info():
    """Returns the Whisper model config so the frontend can display it."""
    wcfg = resolve_whisper_config()
    return JSONResponse({
        "engine":               "local-whisper",
        "model":                wcfg["model"],
        "device":               wcfg["device"],
        "compute_type":         wcfg["compute_type"],
        "language":             wcfg["language"] or "auto",
        "streaming":            True,
        "chunk_interval_secs":  INTERIM_INTERVAL_SECS,
    })


@app.post("/api/vc/session")
async def create_vc_session():
    """Creates and returns a new unique session ID for a pitch session."""
    session_id = new_session()
    logger.info(f"[VC] New pitch session created: {session_id}")
    return JSONResponse({"session_id": session_id})


# ── WebSocket routes ──────────────────────────────────────────────────────────

app.add_api_websocket_route("/ws",        stt_websocket_endpoint)
app.add_api_websocket_route("/ws/vc",     vc_websocket_endpoint)
app.add_api_websocket_route("/ws/vision", vision_websocket_endpoint)