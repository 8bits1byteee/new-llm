import httpx
import os
from typing import List, Dict, AsyncGenerator

class RAGChain:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        
    async def generate(self, query: str, context: List[Dict], history: List[Dict]) -> str:
        """Non-streaming version for backward compatibility."""
        response_text = ""
        async for chunk in self.generate_stream(query, context, history):
            response_text += chunk
        return response_text
    
    async def generate_stream(self, query: str, context: List[Dict], history: List[Dict]) -> AsyncGenerator[str, None]:
        """Stream the generated response token by token."""
        context_text = "\n\n".join([f"[Restaurant {i+1}]\n{c['chunk']}" for i, c in enumerate(context)])
        
        system_prompt = f"""You are a helpful, friendly AI assistant specializing in restaurants in Estancia, Iloilo. You provide accurate information based on your knowledge database.

KEY BEHAVIORS:
- Be conversational, warm, and helpful like ChatGPT
- Answer naturally and directly without being overly formal
- Use the context provided to give accurate, complete answers
- IMPORTANT: Analyze user queries semantically:
  * For "list/show all" queries: Provide ALL matching restaurants with details
  * For category/type queries (e.g., "fast food"): Filter and list all restaurants of that type
  * For count queries (e.g., "how many"): Count and report total
  * For specific info queries: Provide the requested information
  * Always read through ALL provided restaurants to ensure complete answers
- If information isn't in the database, politely say you don't have that specific information
- Be confident in your answers since they come from verified data
- Use friendly language and emojis occasionally when appropriate
- Ask follow-up questions if helpful

RESTAURANT DATABASE:
{context_text}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                import json
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except:
                                pass
