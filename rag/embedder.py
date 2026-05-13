import httpx
import os
from typing import List

class JinaEmbedder:
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        self.url = "https://api.jina.ai/v1/embeddings"
        
    async def embed(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": texts, "model": "jina-embeddings-v5-text-nano"}
            )
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]
