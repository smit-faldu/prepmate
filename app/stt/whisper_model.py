"""
app/stt/whisper_model.py — DEPRECATED singleton model loader.

This module is no longer used.

Previously it held a global faster-whisper WhisperModel singleton that was
passed explicitly to the custom StreamingWhisperProcessor.

After the rewrite to Pipecat's native WhisperSTTService, model loading is
managed internally by the service — no explicit model object needs to be
created or passed around.  The service lazily loads the model on its first
inference call and caches it for the lifetime of the service instance.

This file is kept for reference only and may be deleted.
"""
