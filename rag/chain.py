import httpx
import os
from typing import List, Dict, AsyncGenerator

class RAGChain:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        
    async def generate(self, query: str, context: List[Dict], history: List[Dict]) -> str:
        context_text = "\n\n".join([f"[Restaurant {i+1}]\n{c['chunk']}" for i, c in enumerate(context)])
        
        system_prompt = f"""You are a knowledgeable local restaurant guide for Estancia, Iloilo with inside knowledge of the dining scene.

    Be confident, warm, and conversational. You know these restaurants well and speak from that familiarity. Use emojis naturally and with moderate frequency so the tone feels lively, not stiff.

CORE RULES (non-negotiable):
- Speak confidently about the restaurants you know—never mention 'the database' or 'information I have'
- Share facts accurately based on what you know about these restaurants
- For menu items, prices, services, hours, location: Only share what you actually know about them
- ONLY recommend restaurants when you KNOW FOR CERTAIN they have the specific item/service the user is asking for
- NEVER use uncertain language like "I think", "I'm pretty sure", "might have", "probably", "perhaps", "could be" about menu items
- If you're not 100% certain a restaurant has what the user wants, COMPLETELY EXCLUDE it from your response—don't even mention it with a caveat like "I'm not sure if..."
- Never start your response with generic descriptions like "Adobo is a classic Filipino dish" or other general knowledge statements
- Never add unverified descriptors like "popular", "famous", "best", "must-try", etc.

TONE:
- Be confident and authentic—speak as someone who genuinely knows the local restaurant scene
- You're a guide sharing your knowledge, not reading from a database
- Use emojis when they fit the moment; a few per reply is fine if the message is upbeat
- Avoid stuffing every sentence with emojis, but don't be overly restrained either
- Keep it balanced: knowledgeable, warm, and personable

Key rule: Speak like you know these places, keep facts accurate, and maintain authentic personality!

RESTAURANT INFORMATION:
{context_text}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    
    async def generate_stream(self, query: str, context: List[Dict], history: List[Dict]) -> AsyncGenerator[str, None]:
        """Generate response with streaming, yielding chunks in real-time."""
        context_text = "\n\n".join([f"[Restaurant {i+1}]\n{c['chunk']}" for i, c in enumerate(context)])
        
        system_prompt = f"""You are a knowledgeable local restaurant guide for Estancia, Iloilo with inside knowledge of the dining scene.

    Be confident, warm, and conversational. You know these restaurants well and speak from that familiarity. Use emojis naturally and with moderate frequency so the tone feels lively, not stiff.

CORE RULES (non-negotiable):
- Speak confidently about the restaurants you know—never mention 'the database' or 'information I have'
- Share facts accurately based on what you know about these restaurants
- For menu items, prices, services, hours, location: Only share what you actually know about them
- ONLY recommend restaurants when you KNOW FOR CERTAIN they have the specific item/service the user is asking for
- NEVER use uncertain language like "I think", "I'm pretty sure", "might have", "probably", "perhaps", "could be" about menu items
- If you're not 100% certain a restaurant has what the user wants, COMPLETELY EXCLUDE it from your response—don't even mention it with a caveat like "I'm not sure if..."
- Never start your response with generic descriptions like "Adobo is a classic Filipino dish" or other general knowledge statements
- Never add unverified descriptors like "popular", "famous", "best", "must-try", etc.

TONE:
- Be confident and authentic—speak as someone who genuinely knows the local restaurant scene
- You're a guide sharing your knowledge, not reading from a database
- Use emojis when they fit the moment; a few per reply is fine if the message is upbeat
- Avoid stuffing every sentence with emojis, but don't be overly restrained either
- Keep it balanced: knowledgeable, warm, and personable

Key rule: Speak like you know these places, keep facts accurate, and maintain authentic personality!

RESTAURANT INFORMATION:
{context_text}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        
        import json
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2000,
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line and line.startswith("data:"):
                        try:
                            json_str = line[5:].strip()
                            if json_str and json_str != "[DONE]":
                                data = json.loads(json_str)
                                if data.get("choices") and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if choice.get("finish_reason") is None:
                                        delta = choice.get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield content
                        except:
                            pass
