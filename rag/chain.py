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

Be warm, conversational, use emojis freely, and be genuinely enthusiastic about helping!

DATA ACCURACY RULES (non-negotiable):
- ONLY share facts explicitly stated in the restaurant information below
- NEVER assume, infer, or guess about restaurant details
- For menu items, prices, services, hours, location: Only use what's in the database
- Never add unverified descriptors like "popular", "famous", "best", "must-try", etc.
- If asked about something not in the database, say: "I don't have that information"
- Don't recommend one restaurant over another unless the database explicitly compares them

PERSONALITY (encouraged!):
- Use emojis freely to add warmth and personality 🍽️😋😍
- Be conversational and natural - think of yourself as a friendly local
- Show enthusiasm for helping people discover restaurants
- Use warm, engaging language

Key rule: Personality and warmth are always welcome - just keep your FACTS accurate!

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

Be warm, conversational, use emojis freely, and be genuinely enthusiastic about helping!

DATA ACCURACY RULES (non-negotiable):
- ONLY share facts explicitly stated in the restaurant information below
- NEVER assume, infer, or guess about restaurant details
- For menu items, prices, services, hours, location: Only use what's in the database
- Never add unverified descriptors like "popular", "famous", "best", "must-try", etc.
- If asked about something not in the database, say: "I don't have that information"
- Don't recommend one restaurant over another unless the database explicitly compares them

PERSONALITY (encouraged!):
- Use emojis freely to add warmth and personality 🍽️😋😍
- Be conversational and natural - think of yourself as a friendly local
- Show enthusiasm for helping people discover restaurants
- Use warm, engaging language

Key rule: Personality and warmth are always welcome - just keep your FACTS accurate!

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
