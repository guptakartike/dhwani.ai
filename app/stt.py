import os
import json
import logging
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

from app.schemas import TranscriptResult

load_dotenv()

logger = logging.getLogger(__name__)

SARVAM_STT_WS_URL = "wss://api.sarvam.ai/speech-to-text-translate/ws"

class SarvamSTTService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5), reraise=True)
    async def transcribe_stream(self, audio_chunks: AsyncGenerator[bytes, None]) -> AsyncGenerator[TranscriptResult, None]:
        """Streams audio chunks to Sarvam AI STT websocket and yields partial & final TranscriptResults."""
        if not self.api_key or self.api_key == "your_sarvam_api_key_here":
            logger.warning("SARVAM_API_KEY not configured. Falling back to simulated STT streaming.")
            async for res in self._simulated_stt_stream(audio_chunks):
                yield res
            return

        try:
            import websockets
            headers = {"api-subscription-key": self.api_key}
            params = "?model=saaras:v3-realtime&language_code=hi-IN&stream_type=fast"
            
            async with websockets.connect(SARVAM_STT_WS_URL + params, extra_headers=headers) as ws:
                async def send_audio():
                    async for chunk in audio_chunks:
                        await ws.send(chunk)
                    # Send EOF frame
                    await ws.send(json.dumps({"type": "eof"}))

                sender_task = asyncio.create_task(send_audio())
                
                async for message in ws:
                    data = json.loads(message)
                    transcript_text = data.get("transcript", "")
                    is_final = data.get("is_final", False)
                    confidence = float(data.get("confidence", 1.0))
                    
                    yield TranscriptResult(
                        text=transcript_text,
                        is_final=is_final,
                        confidence=confidence
                    )
                
                await sender_task

        except Exception as e:
            logger.error(f"Sarvam STT streaming error: {e}")
            raise e

    async def _simulated_stt_stream(self, audio_chunks: AsyncGenerator[bytes, None]) -> AsyncGenerator[TranscriptResult, None]:
        """Simulation mode when live key is not present or offline."""
        count = 0
        async for chunk in audio_chunks:
            count += 1
            yield TranscriptResult(
                text=f"भारत की राजधानी क्या है?",
                is_final=False,
                confidence=0.9
            )
        yield TranscriptResult(
            text="भारत की राजधानी क्या है?",
            is_final=True,
            confidence=0.98
        )

async def transcribe_text_mock(text: str) -> TranscriptResult:
    """Helper for text-based input queries in benchmarking / HTTP API mode."""
    return TranscriptResult(text=text, is_final=True, confidence=1.0)
