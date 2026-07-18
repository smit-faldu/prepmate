"""
main.py — PrepMate server entry point.

Start with:
    uvicorn main:app --reload --host 127.0.0.1 --port 8000

Or run directly:
    python main.py
"""

from app.server import app  # noqa: F401 — re-export for uvicorn

if __name__ == "__main__":
    import uvicorn
    from loguru import logger

    logger.info("Starting PrepMate server on http://localhost:8000")
    logger.info("  VC Pitch Arena (default): http://localhost:8000/")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
