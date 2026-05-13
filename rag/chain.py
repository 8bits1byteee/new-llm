import httpx
import os
from typing import List, Dict

class RAGChain:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        
    async def generate(self, query: str, context: List[Dict], history: List[Dict], is_full_list: bool) -> str:
        context_text = "\n\n".join([f"[Restaurant {i+1}]\n{c['chunk']}" for i, c in enumerate(context)])
        
        system_prompt = f"""You are a knowledgeable restaurant assistant for Estancia, Iloilo. You provide accurate, complete information based ONLY on the database provided.

CRITICAL RULES:
1. Answer ONLY from the provided context - never make up information
2. {"Provide a COMPLETE list of ALL matching restaurants" if is_full_list else "Provide detailed information about the most relevant restaurants"}
3. Be conversational and natural like Claude AI
4. If information is missing from context, say "I don't have that information in my database"
5. Always be 100% confident in your answers since they come from verified data
6. Format lists clearly with restaurant names, categories, and key details

RESTAURANT DATABASE:
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
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
