import asyncio
import time
import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO
from hsemotion.facial_emotions import HSEmotionRecognizer

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, UserImageRawFrame, LLMMessagesAppendFrame, TransportMessageFrame, OutputTransportMessageFrame
import torch

class MultimodalVisionProcessor(FrameProcessor):
    def __init__(self, target_fps: float = 3.0, yolo_model: str = "yolo11n-pose.pt"):
        super().__init__()
        self.target_fps = target_fps
        self.min_frame_interval = 1.0 / target_fps
        self.last_process_time = 0
        self.last_state = None

        logger.info("Loading Vision Models (ONNX/TensorRT)...")
        
        self.yolo = YOLO(yolo_model, task='pose')
        
        # --- FIX: Temporarily disable weights_only for PyTorch 2.6+ ---
        # We temporarily patch torch.load so hsemotion can load its model, 
        # avoiding the need to manually allow-list dozens of internal timm classes.
        original_load = torch.load
        
        def legacy_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
            
        try:
            torch.load = legacy_load
            self.emotion_recognizer = HSEmotionRecognizer(
                model_name='enet_b0_8_best_vgaf', 
                device='cpu' # or 'cpu' if you don't have Nvidia GPU
            )
        finally:
            # Always restore the original secure load function immediately after
            torch.load = original_load
            
        logger.success("Vision Models loaded successfully.")
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Pass the raw frame down the pipeline immediately so video isn't blocked
        await super().process_frame(frame, direction)

        # Intercept incoming video frames from the user
        if isinstance(frame, UserImageRawFrame):
            current_time = time.time()
            
            # FPS Throttling: Drop frame if processed too recently
            if current_time - self.last_process_time < self.min_frame_interval:
                return

            self.last_process_time = current_time

            # Offload heavy CV inference to a background thread to keep asyncio unblocked
            state_message = await asyncio.to_thread(self._analyze_frame, frame)

            if state_message:
                # Always send to frontend for realtime updates
                msg = {"type": "vision_update", "text": state_message}
                await self.push_frame(OutputTransportMessageFrame(message=msg), direction)

            # If we detected a valid state and it changed from the last known state
            if state_message and state_message != self.last_state:
                self.last_state = state_message
                logger.info(f"👀 Vision State Changed: {state_message}")
                
                # Inject silent visual context into the pipeline.
                # We format this as a system-level context update.
                context_frame = LLMMessagesAppendFrame(
                    messages=[{
                        "role": "system", 
                        "content": f"[VISUAL CONTEXT UPDATE - DO NOT READ ALOUD: {state_message}]"
                    }]
                )
                await self.push_frame(context_frame, direction)

    def _analyze_frame(self, frame: UserImageRawFrame) -> str | None:
        try:
            # 1. Convert raw bytes to OpenCV format (WebRTC usually outputs RGB or BGRA)
            img_array = np.frombuffer(frame.image, dtype=np.uint8)
            width, height = frame.size
            
            # Reshape based on standard 3-channel RGB
            if len(img_array) == width * height * 3:
                img = img_array.reshape((height, width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                return None # Drop unsupported formats gracefully
            
            # 2. YOLOv11 Pose Detection (Forced on CPU)
            results = self.yolo(img, verbose=False, conf=0.5, device="cpu")
            
            if not results or len(results[0].boxes) == 0:
                return "User is not visible on camera."

            # Get the first detected person
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            keypoints = results[0].keypoints.xy[0].cpu().numpy()

            # 3. Emotion Detection (Face Cropping via Heuristics/Keypoints)
            emotion_str = "neutral"
            if len(keypoints) > 0:
                nose_x, nose_y = keypoints[0]
                if nose_x > 0 and nose_y > 0:
                    # Crop roughly 40% of the bounding box width around the nose
                    face_size = int((x2 - x1) * 0.4)
                    fx1, fy1 = max(0, int(nose_x - face_size/2)), max(0, int(nose_y - face_size/2))
                    fx2, fy2 = min(width, int(nose_x + face_size/2)), min(height, int(nose_y + face_size/2))
                    
                    face_crop = img[fy1:fy2, fx1:fx2]
                    
                    if face_crop.size > 0:
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        emotion, scores = self.emotion_recognizer.predict_emotions(face_rgb)
                        emotion_str = emotion

            # 4. Gesture Detection (Wrist vs Elbow/Shoulder Y-Coordinates)
            gesture_str = "hands are down"
            if len(keypoints) > 10:
                l_shoulder_y = keypoints[5][1]
                r_shoulder_y = keypoints[6][1]
                l_elbow_y = keypoints[7][1]
                r_elbow_y = keypoints[8][1]
                l_wrist_y = keypoints[9][1]
                r_wrist_y = keypoints[10][1]

                # In OpenCV, Y=0 is the top of the image.
                # If wrists are higher than the elbows, the forearms are raised (chest-level gesturing)
                l_gesturing = (l_wrist_y > 0 and l_elbow_y > 0 and l_wrist_y < l_elbow_y)
                r_gesturing = (r_wrist_y > 0 and r_elbow_y > 0 and r_wrist_y < r_elbow_y)
                
                # If wrists are higher than shoulders, hands are way up
                l_high = (l_wrist_y > 0 and l_shoulder_y > 0 and l_wrist_y < l_shoulder_y)
                r_high = (r_wrist_y > 0 and r_shoulder_y > 0 and r_wrist_y < r_shoulder_y)

                if l_high or r_high:
                    gesture_str = "raising their hands high"
                elif l_gesturing or r_gesturing:
                    gesture_str = "actively gesturing with hands"

            return f"User is visibly {emotion_str} and their {gesture_str}."

        except Exception as e:
            logger.error(f"Vision inference error: {e}")
            return None