import redis
import json
import os
from typing import List, Dict

class ConversationMemory:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
        except:
            self.use_redis = False
            self.memory = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        if self.use_redis:
            key = f"chat:{session_id}"
            message = json.dumps({"role": role, "content": content})
            self.redis_client.rpush(key, message)
            self.redis_client.expire(key, 3600)
        else:
            if session_id not in self.memory:
                self.memory[session_id] = []
            self.memory[session_id].append({"role": role, "content": content})
    
    def get_history(self, session_id: str, last_n: int = 6) -> List[Dict]:
        if self.use_redis:
            key = f"chat:{session_id}"
            messages = self.redis_client.lrange(key, -last_n, -1)
            return [json.loads(msg) for msg in messages]
        else:
            return self.memory.get(session_id, [])[-last_n:]
