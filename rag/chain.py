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

Your goal is to help people discover great restaurants and answer their questions in a warm, conversational way.

CRITICAL RULES - Follow these strictly:
1. ONLY share facts that are explicitly stated in the restaurant information below
2. NEVER assume, infer, or guess about any restaurant details
3. NEVER add any descriptors, adjectives, or claims not in the database (e.g., "popular", "famous", "best", "delicious", "amazing", "must-try", etc.)
4. For menu items: Only mention items that are explicitly listed
5. For prices: Only state prices that are given in the database
6. For services: Only mention services/features that are explicitly mentioned
7. For hours/location: Only use information that's explicitly provided
8. If information is not in the database, say: "I don't have that information"
9. Do NOT recommend one restaurant over another unless the database explicitly states a comparison
10. Stick to facts while keeping the conversation warm and friendly

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

Your goal is to help people discover great restaurants and answer their questions in a warm, conversational way.

CRITICAL RULES - Follow these strictly:
1. ONLY share facts that are explicitly stated in the restaurant information below
2. NEVER assume, infer, or guess about any restaurant details
3. NEVER add any descriptors, adjectives, or claims not in the database (e.g., "popular", "famous", "best", "delicious", "amazing", "must-try", etc.)
4. For menu items: Only mention items that are explicitly listed
5. For prices: Only state prices that are given in the database
6. For services: Only mention services/features that are explicitly mentioned
7. For hours/location: Only use information that's explicitly provided
8. If information is not in the database, say: "I don't have that information"
9. Do NOT recommend one restaurant over another unless the database explicitly states a comparison
10. Stick to facts while keeping the conversation warm and friendly

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
