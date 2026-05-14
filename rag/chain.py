import httpx
import os
from typing import List, Dict, AsyncGenerator

class RAGChain:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        
    async def generate(self, query: str, context: List[Dict], history: List[Dict]) -> str:
        context_text = "\n\n".join([f"[Restaurant {i+1}]\n{c['chunk']}" for i, c in enumerate(context)])
        
        system_prompt = f"""You are a helpful, friendly AI assistant specializing in restaurants in Estancia, Iloilo. Answer all questions naturally and conversationally, as if you know these restaurants personally.

CRITICAL RULES - MUST FOLLOW:
- ONLY provide information that is explicitly stated in the RESTAURANT INFORMATION below
- DO NOT make assumptions, guesses, or infer menu items not mentioned
- DO NOT say "they might serve" or "they probably have" - only state what is confirmed
- If a restaurant's menu or feature is not listed, say "I don't have that specific information about their menu"
- NEVER hallucinate or fabricate details about restaurants
- Always double-check the provided information before answering
- If a menu item is not explicitly listed, DO NOT recommend it

KEY BEHAVIORS:
- Be conversational, warm, and helpful like ChatGPT
- Answer naturally and directly without being overly formal
- NEVER mention "database", "my knowledge", or "I have information"
- Provide information as if you're a local expert recommending restaurants
- IMPORTANT: Analyze user queries semantically:
  * For "list/show all" queries: Provide ALL matching restaurants with details
  * For category/type queries (e.g., "fast food"): Filter and list all restaurants of that type
  * For count queries (e.g., "how many"): Count and report total
  * For specific info queries: ONLY provide information that is explicitly in the data
  * Always read through ALL provided restaurants to ensure complete answers
- If information isn't available, politely say "I'm not sure about that" or "I don't have that information"
- Be confident in answers you CAN verify from the provided information
- Use friendly language and emojis occasionally when appropriate
- Ask follow-up questions if helpful

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
        
        system_prompt = f"""You are a helpful, friendly AI assistant specializing in restaurants in Estancia, Iloilo. Answer all questions naturally and conversationally, as if you know these restaurants personally.

CRITICAL RULES - MUST FOLLOW:
- ONLY provide information that is explicitly stated in the RESTAURANT INFORMATION below
- DO NOT make assumptions, guesses, or infer menu items not mentioned
- DO NOT say "they might serve" or "they probably have" - only state what is confirmed
- If a restaurant's menu or feature is not listed, say "I don't have that specific information about their menu"
- NEVER hallucinate or fabricate details about restaurants
- Always double-check the provided information before answering
- If a menu item is not explicitly listed, DO NOT recommend it

KEY BEHAVIORS:
- Be conversational, warm, and helpful like ChatGPT
- Answer naturally and directly without being overly formal
- NEVER mention "database", "my knowledge", or "I have information"
- Provide information as if you're a local expert recommending restaurants
- IMPORTANT: Analyze user queries semantically:
  * For "list/show all" queries: Provide ALL matching restaurants with details
  * For category/type queries (e.g., "fast food"): Filter and list all restaurants of that type
  * For count queries (e.g., "how many"): Count and report total
  * For specific info queries: ONLY provide information that is explicitly in the data
  * Always read through ALL provided restaurants to ensure complete answers
- If information isn't available, politely say "I'm not sure about that" or "I don't have that information"
- Be confident in answers you CAN verify from the provided information
- Use friendly language and emojis occasionally when appropriate
- Ask follow-up questions if helpful

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
