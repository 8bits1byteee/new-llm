import pandas as pd
import asyncio
import json
from rag.embedder import JinaEmbedder
from dotenv import load_dotenv

load_dotenv()

async def ingest_csv():
    df = pd.read_csv("restaurants (7).csv")
    
    chunks = []
    for _, row in df.iterrows():
        chunk = f"""Restaurant: {row['Restaurant Name']}
Category: {row['Category']}
Description: {row['Description']}
Address: {row['Address']}
Phone: {row['Phone']}
Email: {row['Email']}
Hours: {row['Operating Hours']}
Services: {row['Service Options']}
Price Range: {row['Price Range']}
Menu Highlights: {row['Menu']}
Capacity: {row['Seating Capacity']}
Payment: {row['Payment Methods']}
Accessibility: {row['Accessibility']}"""
        chunks.append(chunk)
    
    embedder = JinaEmbedder()
    print(f"Embedding {len(chunks)} restaurants...")
    embeddings = await embedder.embed(chunks)
    
    with open("data/vector_store.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "embeddings": embeddings}, f)
    
    print("✓ Vector store created successfully!")

if __name__ == "__main__":
    asyncio.run(ingest_csv())
