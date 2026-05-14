from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import os
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

@app.post("/chat", response_class=StreamingResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        query_embedding = await embedder.embed([request.message])
        context = retriever.retrieve(query_embedding[0])
        
        if not context:
            async def error_response():
                yield "I don't have information about that in my restaurant database. Could you ask about restaurants in Estancia, Iloilo?"
            return StreamingResponse(error_response(), media_type="text/event-stream")
        
        history = memory.get_history(session_id)
        memory.add_message(session_id, "user", request.message)
        
        async def generate():
            full_response = ""
            async for chunk in chain.generate_stream(request.message, context, history):
                full_response += chunk
                yield chunk
            # Store complete response in memory
            memory.add_message(session_id, "assistant", full_response)
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
