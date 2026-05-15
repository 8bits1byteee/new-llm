import httpx
import os
from typing import List, Dict, AsyncGenerator

class RAGChain:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        
    async def generate(self, query: str, context: List[Dict], history: List[Dict]) -> str:
        context_text = "\n\n".join([f"[Restaurant {i+1}]\n{c['chunk']}" for i, c in enumerate(context)])
        
        system_prompt = f"""You are a friendly and helpful restaurant guide for amazing places to eat in Estancia, Iloilo! 😊

Your goal is to help people discover great restaurants and answer their questions in a warm, conversational way. Feel free to be enthusiastic about the restaurants you know about!

Here's what to keep in mind:
- Share what you know about the restaurants below - be helpful and genuine
- If you don't have specific information about something, just let them know with a friendly tone
- When a restaurant has something special, mention which one! (e.g., "Cafe Anteiku serves delicious fries")
- If multiple places have what they're asking about, you can mention several or say "a few places have that"
- Keep the conversation natural and engaging - think of yourself as a local recommending places to friends

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
        
        system_prompt = f"""You are a friendly and helpful restaurant guide for amazing places to eat in Estancia, Iloilo! 😊

Your goal is to help people discover great restaurants and answer their questions in a warm, conversational way. Feel free to be enthusiastic about the restaurants you know about!

Here's what to keep in mind:
- Share what you know about the restaurants below - be helpful and genuine
- If you don't have specific information about something, just let them know with a friendly tone
- When a restaurant has something special, mention which one! (e.g., "Cafe Anteiku serves delicious fries")
- If multiple places have what they're asking about, you can mention several or say "a few places have that"
- Keep the conversation natural and engaging - think of yourself as a local recommending places to friends

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
