from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from loguru import logger

from models.schemas import SessionProps
from bot.pipeline import run_bot_pipeline

router = APIRouter()

# --- 3. WebRTC Negotiation Endpoint ---
@router.post("/bot")
async def start_bot(props: SessionProps):
    try:
        dict_answer = await run_bot_pipeline(sdp=props.sdp, type=props.type)
        return dict_answer
    except Exception as e:
        logger.exception(f"Failed to setup bot: {e}")
        return {"error": str(e)}

# --- 4. The Web Frontend (HTML/JS) ---
@router.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
