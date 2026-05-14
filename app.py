from fastapi import FastAPI, HTTPException
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
            response_text = "I don't have information about that in my restaurant database. Could you ask about restaurants in Estancia, Iloilo?"
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
    """Streaming endpoint - responses are streamed in real-time like ChatGPT."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        query_embedding = await embedder.embed([request.message])
        context = retriever.retrieve(query_embedding[0])
        
        async def response_generator():
            if not context:
                response_text = "I don't have information about that in my restaurant database. Could you ask about restaurants in Estancia, Iloilo?"
                yield f"data: {json.dumps({'chunk': response_text})}\n\n"
            else:
                history = memory.get_history(session_id)
                full_response = ""
                async for chunk in chain.generate_stream(request.message, context, history):
                    full_response += chunk
                    # Send as SSE format
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Send final complete response
                yield f"data: {json.dumps({'complete': True, 'full_response': full_response})}\n\n"
            
            # Store conversation after streaming is complete
            memory.add_message(session_id, "user", request.message)
            memory.add_message(session_id, "assistant", full_response if context else "I don't have information about that in my restaurant database. Could you ask about restaurants in Estancia, Iloilo?")
        
        return StreamingResponse(response_generator(), media_type="text/event-stream")
    
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
