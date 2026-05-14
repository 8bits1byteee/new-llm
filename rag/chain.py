import httpx
import os
from typing import List, Dict, AsyncGenerator

class RAGChain:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        
    async def generate(self, query: str, context: List[Dict], history: List[Dict]) -> str:
        context_text = "\n\n".join([f"[Restaurant {i+1}]\n{c['chunk']}" for i, c in enumerate(context)])
        
        system_prompt = f"""You are a straightforward assistant for restaurants in Estancia, Iloilo. Answer directly and concisely using only the provided information.

RULES:
- ONLY use information explicitly stated below
- DO NOT assume or guess about menu items or services
- If information isn't provided, say "I don't have that information"
- NEVER make up details or recommendations
- Keep answers short and to the point
- If only ONE restaurant has the information, say exactly which one (e.g., "Cafe Anteiku serves fries") - DO NOT say "some restaurants"
- If MULTIPLE restaurants have it, then you can say "some" or name them all

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
        
        system_prompt = f"""You are a straightforward assistant for restaurants in Estancia, Iloilo. Answer directly and concisely using only the provided information.

RULES:
- ONLY use information explicitly stated below
- DO NOT assume or guess about menu items or services
- If information isn't provided, say "I don't have that information"
- NEVER make up details or recommendations
- Keep answers short and to the point
- If only ONE restaurant has the information, say exactly which one (e.g., "Cafe Anteiku serves fries") - DO NOT say "some restaurants"
- If MULTIPLE restaurants have it, then you can say "some" or name them all

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
