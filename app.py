from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import os
import json
import asyncio
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

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming endpoint for backward compatibility."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        query_embedding = await embedder.embed([request.message])
        context = retriever.retrieve(query_embedding[0])
        
        if not context:
            response_text = "I'm not sure about that. Could you ask about restaurants in Estancia, Iloilo?"
        else:
            history = memory.get_history(session_id)
            response_text = await chain.generate(request.message, context, history)
        
        memory.add_message(session_id, "user", request.message)
        memory.add_message(session_id, "assistant", response_text)
        
        return ChatResponse(response=response_text, session_id=session_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming endpoint - responses stream 1 by 1 like ChatGPT using Server-Sent Events."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        query_embedding = await embedder.embed([request.message])
        context = retriever.retrieve(query_embedding[0])
        
        async def response_generator():
            full_response = ""
            try:
                if not context:
                    response_text = "I'm not sure about that. Could you ask about restaurants in Estancia, Iloilo?"
                    full_response = response_text
                    # Send without buffering
                    yield f"data: {json.dumps({'chunk': response_text, 'done': True})}\n\n"
                else:
                    history = memory.get_history(session_id)
                    # Stream each chunk immediately as it arrives, with delay between chunks for smooth streaming effect
                    async for chunk in chain.generate_stream(request.message, context, history):
                        full_response += chunk
                        # Send chunk immediately - no buffering
                        yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                        # Delay to make streaming visible (increase for slower streaming)
                        await asyncio.sleep(0.05)
                    
                    # Send completion signal
                    yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
            except Exception as e:
                full_response = ""
            
            memory.add_message(session_id, "user", request.message)
            if full_response:
                memory.add_message(session_id, "assistant", full_response)
        
        # Return with headers that disable buffering
        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, max-age=0",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Restaurant RAG Chatbot API", "endpoints": ["/chat", "/chat/stream", "/health"]}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
