# Restaurant RAG Chatbot

Industry-level RAG chatbot for Estancia, Iloilo restaurants with context awareness and conversational AI.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```
GROQ_API_KEY=your_key
JINA_API_KEY=your_key
REDIS_URL=redis://localhost:6379
```

3. Generate embeddings:
```bash
python ingest.py
```

4. Run server:
```bash
uvicorn app:app --reload
```

## API Endpoints

**POST /chat**
```json
{
  "message": "Show me all cafes",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "response": "Here are all the cafes...",
  "session_id": "uuid"
}
```

## Railway Deployment

1. Connect GitHub repo to Railway
2. Add environment variables in Railway dashboard
3. Deploy automatically on push

## Features

- ✅ Context-aware conversations with Redis memory
- ✅ Full-list detection for complete answers
- ✅ 100% confidence from RAG retrieval
- ✅ Claude-like conversational style
- ✅ No hallucinations - answers only from database
