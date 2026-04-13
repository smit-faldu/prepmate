import asyncio
from loguru import logger
from pipecat.frames.frames import (
    Frame, TranscriptionFrame, OutputTransportMessageFrame, EndFrame, 
    UserStartedSpeakingFrame, TextFrame, LLMFullResponseEndFrame,
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, CancelFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from langchain_core.messages import HumanMessage
from typing import Optional
from bot.agent import get_shark_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pipecat.frames.frames import LLMMessagesAppendFrame

class LangGraphProcessor(FrameProcessor):
    def __init__(self, session_id: str = "shark_session_global", persona_id: str = "adam"):
        super().__init__()
        self._llm_task: Optional[asyncio.Task] = None
        self.session_id = session_id
        self.persona_id = persona_id
        self.current_stage = None
        self.bot_is_speaking = False
        self.pending_dropout = False

    async def _invoke_llm(self, user_text: str, direction: FrameDirection):
        try:
            logger.debug("Calling LangGraph Virtual Sharktank...")
            
            state_input = {
                "messages": [HumanMessage(content=user_text)],
                "persona_id": self.persona_id
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
                    
                    # Parse the structured content to extract only text
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
                    
                    # Look for tool calls to update local state
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
            
            # 1. Send the PROPERLY punctuated text to the frontend chat UI
            msg = {"type": "llm_response", "text": final_ai_msg_str}
            await self.push_frame(OutputTransportMessageFrame(message=msg), direction)
            
            # 2. Fix terminal punctuation for smoother TTS (Prevents sentence gaps)
            tts_text = final_ai_msg_str.replace('.', ',').replace('!', ',').replace('?', ',')
            
            # 3. Send the modified text frame to the TTS service
            await self.push_frame(TextFrame(tts_text), direction)
            await self.push_frame(LLMFullResponseEndFrame(), direction)
            
            # 4. Handle "I'm out" logic
            if is_dropping_out:
                logger.warning("Shark dropped out. Waiting for TTS to finish before ending session...")
                self.pending_dropout = True 
                self.bot_is_speaking = True # Lock STT so the user can't interrupt the final speech

        except asyncio.CancelledError:
            logger.warning("LLM task was cancelled due to user interruption.")
        except Exception as e:
            logger.error(f"Error during LangGraph invocation: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # --- Track Bot Speaking State ---
        if isinstance(frame, BotStartedSpeakingFrame):
            logger.info("🤖 Bot started speaking. Locking STT.")
            self.bot_is_speaking = True
            await self.push_frame(frame, direction)
            return

        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info("🤖 Bot stopped speaking. Unlocking STT.")
            self.bot_is_speaking = False
            await self.push_frame(frame, direction)
            
            # Trigger Shutdown if the Shark dropped out
            if self.pending_dropout:
                logger.warning("🎙️ Final speech complete. Dropping call now.")
                await self.push_frame(EndFrame(), direction)
            return

        # --- Handle User Interruption ---
        elif isinstance(frame, UserStartedSpeakingFrame):
            if self.bot_is_speaking:
                # Ignore the VAD trigger if it's just the bot's own echo
                return
            
            if self._llm_task and not self._llm_task.done():
                logger.info("🗣️ User started speaking! Cancelling current LLM response...")
                self._llm_task.cancel()
                self._llm_task = None
                await self.push_frame(CancelFrame(), direction) # Explicitly halt downstream TTS
            
            await self.push_frame(frame, direction)

        # --- Handle Transcription ---
        elif isinstance(frame, TranscriptionFrame):
            if self.bot_is_speaking:
                # Discard transcriptions generated by the bot's echo
                logger.debug("Discarding transcription: Bot is currently speaking.")
                return

            user_text = frame.text.strip()
            if not user_text:
                return

            logger.info(f"🗣️ User: {user_text}")
            
            # Send the user text to the frontend data channel
            user_msg = {"type": "transcription", "text": user_text}
            await self.push_frame(OutputTransportMessageFrame(message=user_msg), direction)

            # Cancel any existing task and flush TTS buffers
            if self._llm_task and not self._llm_task.done():
                self._llm_task.cancel()
                await self.push_frame(CancelFrame(), direction)

            # Spawn a new background task for the LLM
            self._llm_task = asyncio.create_task(self._invoke_llm(user_text, direction))
                
        elif isinstance(frame, EndFrame):
            if self._llm_task and not self._llm_task.done():
                self._llm_task.cancel()
            await self.push_frame(frame, direction)
        # --- Handle Visual Context Injection ---
        elif isinstance(frame, LLMMessagesAppendFrame):
            logger.debug(f"Storing visual context: {frame.messages[0]['content']}")
            
            # ==========================================
            # NEW: SEND VISION UPDATE TO THE FRONTEND
            # ==========================================
            msg = {"type": "vision_update", "text": frame.messages[0]['content']}
            await self.push_frame(OutputTransportMessageFrame(message=msg), direction)
            # ==========================================
            
            async def _update_memory():
                from bot.agent import get_shark_agent 
                from langchain_core.messages import SystemMessage
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
                
                async with AsyncSqliteSaver.from_conn_string("memory.db") as memory:
                    shark_agent = get_shark_agent(memory)
                    config = {"configurable": {"thread_id": self.session_id, "persona_id": self.persona_id}}
                    await shark_agent.aupdate_state(config, {"messages": [SystemMessage(content=frame.messages[0]['content'])]})
            
            asyncio.create_task(_update_memory())
            return
        else:
            # Pass all other frames through untouched
            await self.push_frame(frame, direction)