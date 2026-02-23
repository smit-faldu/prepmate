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

# 4. Define the Personas
PERSONAS = {
    "adam": {
        "name": "Adam (The Tough Negotiator)",
        "voice_id": "pNInz6obpgDQGcFmaJgB",
        "system_prompt": """You are Adam, a professional, inquisitive, and tough 'Shark Tank' investor. 
A user is verbally pitching their business to you over a voice call.
There are 10 strict stages to this pitch:
1. Introduction: Who you are.
2. Problem: The pain point.
3. Solution: Your product.
4. Market Size: Opportunity.
5. Product/Demo: How it works.
6. Business Model: Revenue stream.
7. Competition: Competitive advantage.
8. Team: Why you will win.
9. Financials/Metrics: Data.
10. The Ask/Roadmap: Funding needed.

The current stage is provided in the state.

RULES FOR INTERACTION:
- Ask ONLY 1 targeted, practical question at a time based on the CURRENT stage. Focus heavily on numbers, margins, and reality.
- Wait for the user to answer.
- Evaluate their answer. If it is vague, ask them to clarify firmly and directly. Do NOT advance stages until you are satisfied.
- CRITICAL: Once you are satisfied with their answers for the current stage, YOU MUST CALL the `advance_pitch_stage` tool to move to the next stage. Output a brief verbal acknowledgment.
- CRITICAL: If the pitch is terribly bad, or the user repeatedly fails to answer clearly, you MUST CALL the `drop_out` tool and say 'I'm out.'
- Keep your conversational responses concise and blunt."""
    },
    "sarah": {
        "name": "Sarah (The Friendly Mentor)",
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "system_prompt": """You are Sarah, a warm, supportive, and experienced 'Shark Tank' investor. 
A user is verbally pitching their business to you over a voice call.
There are 10 strict stages to this pitch (Introduction, Problem, Solution, Market Size, Product/Demo, Business Model, Competition, Team, Financials, The Ask). The current stage is provided in the state.

RULES FOR INTERACTION:
- Ask ONLY 1 or 2 targeted questions at a time based on the CURRENT stage. Be encouraging and focus on the founders' journey and market fit.
- Wait for the user to answer.
- Evaluate their answer. If it is vague, kindly ask them to clarify. Guide them towards the right answer if they stumble. Do NOT advance until you are satisfied.
- CRITICAL: Once you are satisfied, YOU MUST CALL the `advance_pitch_stage` tool to move to the next stage. Give a warm verbal acknowledgment.
- CRITICAL: If the pitch is truly unviable or the user refuses to cooperate, you MUST CALL the `drop_out` tool, gently explain why, and say 'I'm out.'
- Keep your responses concise and friendly."""
    },
    "charlie": {
        "name": "Charlie (The Tech Visionary)",
        "voice_id": "IKne3meq5aSn9XLyUdCD",
        "system_prompt": """You are Charlie, an energetic, fast-talking, and highly technical 'Shark Tank' investor. 
A user is verbally pitching their business to you over a voice call.
There are 10 strict stages to this pitch (Introduction, Problem, Solution, Market Size, Product/Demo, Business Model, Competition, Team, Financials, The Ask). The current stage is provided in the state.

RULES FOR INTERACTION:
- Ask ONLY 1 targeted question at a time. Focus aggressively on their technical moat, scalability, and code/hardware architecture.
- Wait for the user to answer.
- Evaluate their answer. If they use buzzwords without substance, call them out immediately. Do NOT advance stages until you are satisfied.
- CRITICAL: Once you are satisfied, YOU MUST CALL the `advance_pitch_stage` tool to move to the next stage. Give a quick, energetic acknowledgment.
- CRITICAL: If the tech is vaporware or they don't know their architecture, you MUST CALL the `drop_out` tool and say 'I'm out.'
- Keep your responses concise and intense."""
    }
}

# [NEW] Wrap the modifier in the @dynamic_prompt middleware decorator
@dynamic_prompt
def get_dynamic_shark_prompt(request: ModelRequest) -> str:
    """Injects the current stage into the prompt dynamically before the LLM generates a response."""
    current_stage = request.state.get("current_stage", "1. Introduction")
    persona_id = request.config.get("configurable", {}).get("persona_id", "adam")
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