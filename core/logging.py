import sys
from loguru import logger

def setup_logging():
    # --- Enhanced Logging Setup ---
    logger.remove()
    logger.add(sys.stderr, level="DEBUG") # Set to DEBUG to see all Pipecat internal events
    return logger
