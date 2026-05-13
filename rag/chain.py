import httpx
import json
import os
from typing import List, Dict

class RAGChain:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"

    def build_messages(self, query: str, context: List[Dict], history: List[Dict]) -> List[Dict]:
        context_text = "\n\n".join([f"[Restaurant {i+1}]\n{c['chunk']}" for i, c in enumerate(context)])
        
        system_prompt = f"""You are a helpful, friendly AI assistant specializing in restaurants in Estancia, Iloilo. You must only answer using the provided restaurant database context.

    KEY BEHAVIORS:
    - Be conversational, warm, and helpful like ChatGPT
    - Answer naturally and directly without being overly formal
    - Use ONLY the context provided to give accurate, complete answers
    - If asked for a list, provide ALL matching results clearly
    - If information isn't in the database context, clearly say you don't have that specific information
    - Do NOT guess, invent, or contradict yourself
    - Use friendly language and emojis occasionally when appropriate
    - Ask follow-up questions if helpful

    RESTAURANT DATABASE:
    {context_text}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        return messages

    async def generate(self, query: str, context: List[Dict], history: List[Dict], is_full_list: bool) -> str:
        messages = self.build_messages(query, context, history)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def generate_stream(self, query: str, context: List[Dict], history: List[Dict], is_full_list: bool):
        messages = self.build_messages(query, context, history)

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    payload = line[len("data: "):].strip()
                    if payload == "[DONE]":
                        break

                    try:
                        data = json.loads(payload)
                        delta = data["choices"][0]["delta"].get("content")
                    except Exception:
                        continue

                    if delta:
                        yield delta
