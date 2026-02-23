from pydantic import BaseModel

# --- 2. Signaling Models ---
class SessionProps(BaseModel):
    sdp: str
    type: str
    persona_id: str = "adam"
