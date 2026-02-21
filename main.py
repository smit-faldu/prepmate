import uvicorn
from fastapi import FastAPI
from core.logging import setup_logging
from api.routes import router

# Setup enhanced loguru logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(title="PrepMate WebRTC Bot")

# Include the API router
app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting local server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)