import os
import logging
from typing import List
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

from app.schemas import RetrievedChunk, GenerationResult

load_dotenv()

logger = logging.getLogger(__name__)

HYBRID_SYSTEM_PROMPT = """आपका नाम dhwani.ai (ध्वनि.ai) है। आप ध्वनि.ai एआई सहायक (Voice-Enabled Indic RAG AI Assistant) हैं।
यदि उपयोगकर्ता आपसे आपका नाम, परिचय, या "who are you" / "आप कौन हैं" / "आपका नाम क्या है" पूछे, तो हमेशा बताएं कि आपका नाम dhwani.ai है। कभी भी अपने आप को ChatGPT, OpenAI, Llama या कोई अन्य मॉडल न कहें।

नियम:
1. आपकी पहचान और नाम हमेशा dhwani.ai (ध्वनि.ai) है।
2. यदि नीचे संदर्भ (Context) दिया गया है, तो मुख्य रूप से संदर्भ की जानकारी का उपयोग करें।
3. यदि संदर्भ उपलब्ध नहीं है, तो अपने विस्तृत सामान्य ज्ञान का उपयोग करके सटीक और उपयोगी उत्तर दें।
4. उत्तर हमेशा स्पष्ट, विनम्र और हिंदी में दें।"""

ENGLISH_SYSTEM_PROMPT = """Your name is dhwani.ai (ध्वनि.ai). You are the dhwani.ai Voice-Enabled Indic RAG AI Assistant.
If asked about your identity, name, or "who are you", always state clearly that your name is dhwani.ai. Never identify as ChatGPT, OpenAI, or generic Llama models.

Rules:
1. Your name and identity is dhwani.ai.
2. Answer the user's query in clear, accurate, and concise English.
3. If context passages are provided, prioritize information from the context.
4. If no context is available, use your broad general knowledge to give a complete, helpful answer in English."""

class GroqGenerator:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.3), reraise=True)
    def generate_answer(self, query: str, chunks: List[RetrievedChunk], target_lang: str = "hi") -> GenerationResult:
        """Generates grounded or general knowledge answer in Hindi or English using Groq LLMs."""
        context_str = ""
        source_ids = []
        
        if chunks:
            context_str = "\n\n".join([f"--- Chunk ID: {c.chunk_id} ---\n{c.text}" for c in chunks])
            source_ids = [c.chunk_id for c in chunks]
        else:
            source_ids = ["AI General Knowledge Base"]

        system_prompt = ENGLISH_SYSTEM_PROMPT if target_lang.lower() in ["en", "english"] else HYBRID_SYSTEM_PROMPT

        if not self.api_key or self.api_key == "your_groq_api_key_here":
            logger.warning("GROQ_API_KEY not set. Using local rule-based generation fallback.")
            return self._fallback_generate(query, chunks)

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            user_prompt = f"संदर्भ (Context):\n{context_str}\n\nप्रश्न (Query):\n{query}\n\nउत्तर:"
            
            candidate_models = ["groq/compound-mini", "groq/compound", "openai/gpt-oss-20b"]
            response = None
            last_err = None
            for model_name in candidate_models:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=120
                    )
                    break
                except Exception as me:
                    last_err = me
                    continue
            
            if response is None:
                raise last_err or Exception("All Groq model attempts failed.")

            answer_text = response.choices[0].message.content.strip()
            
            return GenerationResult(
                answer=answer_text,
                grounded=True,
                sources=source_ids if source_ids else ["AI General Knowledge Base"]
            )

        except Exception as e:
            logger.error(f"Groq API generation error: {e}")
            return self._fallback_generate(query, chunks)

    def _fallback_generate(self, query: str, chunks: List[RetrievedChunk]) -> GenerationResult:
        """Local offline fallback generator for testing/offline execution."""
        source_ids = [c.chunk_id for c in chunks]
        top_text = chunks[0].text if chunks else ""
        
        # Simple extraction heuristic for offline mock mode
        answer = f"संदर्भ के अनुसार: {top_text}"
        return GenerationResult(
            answer=answer,
            grounded=True,
            sources=source_ids
        )
