from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest# from langgraph.checkpoint.sqlite import SqliteSaver
from core.config import settings
# import sqlite3 # Make sure this is imported at the top
# 1. State Definition
class PitchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_stage: str
    persona_id: str

# 2. Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=settings.GEMINI_API_KEY,
    temperature=0.7,
)

# 3. Define the Tools
@tool
def advance_pitch_stage(current_stage: str, next_stage: str) -> str:
    """Call this tool ONLY when the user has satisfactorily answered the questions for the `current_stage`.
    This indicates to the system that the current stage is complete and we are moving to the `next_stage`.
    
    Args:
        current_stage: The name of the stage that was just completed.
        next_stage: The name of the next stage to move to.
    """
    return f"System: Successfully advanced from '{current_stage}' to '{next_stage}'. You may now begin asking questions about the {next_stage}."

@tool
def drop_out(reason: str) -> str:
    """Call this tool when the user's pitch is irredeemably bad, they are entirely unprepared, or they refuse to answer directly after multiple attempts to clarify.
    This effectively ends the session ("I'm out.")
    
    Args:
        reason: The reason why you are dropping out.
    """
    return f"System: You have dropped out. The session is ending."

from bot.persona import PERSONAS

# [NEW] Wrap the modifier in the @dynamic_prompt middleware decorator
@dynamic_prompt
def get_dynamic_shark_prompt(request: ModelRequest) -> str:
    """Injects the current stage into the prompt dynamically before the LLM generates a response."""
    current_stage = request.state.get("current_stage", "1. Introduction")
    persona_id = request.state.get("persona_id", "adam")
    system_prompt = PERSONAS.get(persona_id, PERSONAS["adam"])["system_prompt"]
    
    return f"{system_prompt}\n\n[SYSTEM CONTEXT]\nThe user is currently on stage: {current_stage}."

def get_shark_agent(memory):
    return create_agent(
        model=llm,
        tools=[advance_pitch_stage, drop_out],
        middleware=[get_dynamic_shark_prompt], 
        state_schema=PitchState,
        checkpointer=memory
    )