import asyncio
from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, OutputTransportMessageFrame, EndFrame, UserStartedSpeakingFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any, Optional
from bot.agent import get_shark_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

class LangGraphProcessor(FrameProcessor):
    def __init__(self, session_id: str = "shark_session_global", persona_id: str = "adam"):
        super().__init__()
        # We no longer manually track `self.messages` because the Checkpointer does it.
        self._llm_task: Optional[asyncio.Task] = None
        self.session_id = session_id
        self.persona_id = persona_id
        self.current_stage = None # Initialize current_stage to prevent attribute errors

    async def _invoke_llm(self, user_text: str, direction: FrameDirection):
        try:
            logger.debug("Calling LangGraph Virtual Sharktank...")
            
            state_input = {
                "messages": [HumanMessage(content=user_text)]
            }
            
            config = {"configurable": {"thread_id": self.session_id, "persona_id": self.persona_id}}
            
            # Use the Async context manager to handle the DB connection safely
            async with AsyncSqliteSaver.from_conn_string("memory.db") as memory:
                shark_agent = get_shark_agent(memory)
                response_state = await shark_agent.ainvoke(state_input, config)
            
            final_ai_msg_str = ""
            is_dropping_out = False
            
            for msg in reversed(response_state["messages"]):
                if msg.type == "ai" and msg.content:
                    
                    # --- FIX: Parse the structured content to extract only text ---
                    if isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                final_ai_msg_str += item.get("text", "")
                            elif isinstance(item, str):
                                final_ai_msg_str += item
                    elif isinstance(msg.content, str):
                        final_ai_msg_str = msg.content
                    else:
                        final_ai_msg_str = str(msg.content)
                    
                    final_ai_msg_str = final_ai_msg_str.strip()
                    # -------------------------------------------------------------
                    
                    # Also look for tool calls if any to update our local state
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if tool_call["name"] == "advance_pitch_stage":
                                args = tool_call.get("args", {})
                                if "next_stage" in args:
                                    self.current_stage = args["next_stage"]
                                    logger.info(f"🔄 Stage Advanced to: {self.current_stage}")
                            elif tool_call["name"] == "drop_out":
                                logger.info(f"🦈 Shark is dropping out: {tool_call.get('args', {}).get('reason', 'No reason given')}")
                                is_dropping_out = True
                    
                    break
                    
            if not final_ai_msg_str:
                logger.warning("No text content found in AI response.")
                return
                
            logger.success(f"🦈 Shark: {final_ai_msg_str}")
            msg = {"type": "llm_response", "text": final_ai_msg_str}
            await self.push_frame(OutputTransportMessageFrame(message=msg), direction)
            # Send text frame to the TTS service downstream
            await self.push_frame(TextFrame(final_ai_msg_str), direction)
            
            # If the shark dropped out, forcefully end the session
            if is_dropping_out:
                logger.warning("Terminating session pipeline due to Shark dropping out.")
                # Pushing EndFrame signals Pipecat to cleanup and stop the runner
                await self.push_frame(EndFrame(), direction)

        except asyncio.CancelledError:
            logger.warning("LLM task was cancelled due to user interruption.")
        except Exception as e:
            logger.error(f"Error during LangGraph invocation: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            if self._llm_task and not self._llm_task.done():
                logger.info("🗣️ User interrupted! Cancelling current LLM response...")
                self._llm_task.cancel()
                self._llm_task = None
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            if not user_text:
                return

            logger.info(f"🗣️ User: {user_text}")
            
            # Send the user text to the frontend data channel
            user_msg = {"type": "transcription", "text": user_text}
            await self.push_frame(OutputTransportMessageFrame(message=user_msg), direction)

            # Cancel any existing task just in case
            if self._llm_task and not self._llm_task.done():
                self._llm_task.cancel()

            # Spawn a new background task for the LLM
            self._llm_task = asyncio.create_task(self._invoke_llm(user_text, direction))
                
        elif isinstance(frame, EndFrame):
            # Clean up memory if needed, though this processor instance is 
            # tied to the pipeline lifecycle
            if self._llm_task and not self._llm_task.done():
                self._llm_task.cancel()
            await self.push_frame(frame, direction)
        else:
            # Pass all other frames through untouched
            await self.push_frame(frame, direction)