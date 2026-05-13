from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import os
import json
from rag.embedder import JinaEmbedder
from rag.retriever import Retriever
from rag.chain import RAGChain
from rag.memory import ConversationMemory

load_dotenv()

app = FastAPI(title="Restaurant RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = JinaEmbedder()
retriever = Retriever("data/vector_store.json")
chain = RAGChain()
memory = ConversationMemory()

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

def format_sse_event(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    async def event_stream():
        try:
            query_embedding = await embedder.embed([request.message])
            context, is_full_list = retriever.retrieve(query_embedding[0])

            yield format_sse_event("meta", {"session_id": session_id})

            if not context:
                response_text = (
                    "I don't have information about that in my restaurant database. "
                    "Could you ask about restaurants in Estancia, Iloilo?"
                )
                yield format_sse_event("chunk", response_text)
                memory.add_message(session_id, "user", request.message)
                memory.add_message(session_id, "assistant", response_text)
                yield format_sse_event("done", {"session_id": session_id})
                return

            history = memory.get_history(session_id)
            full_text = ""
            async for chunk in chain.generate_stream(request.message, context, history, is_full_list):
                full_text += chunk
                yield format_sse_event("chunk", chunk)

            memory.add_message(session_id, "user", request.message)
            memory.add_message(session_id, "assistant", full_text)
            yield format_sse_event("done", {"session_id": session_id})

        except Exception as e:
            yield format_sse_event("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/")
async def root():
    return {"message": "Restaurant RAG Chatbot API", "endpoints": ["/chat", "/health"]}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
