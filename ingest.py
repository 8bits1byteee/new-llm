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
Facebook Name: {row['Facebook Name']}
Facebook URL: {row['Facebook URL']}
Operating Hours: {row['Operating Hours']}
Service Options: {row['Service Options']}
Reservation: {row['Reservation']}
Price Range: {row['Price Range']}
Menu: {row['Menu']}
Seating Capacity: {row['Seating Capacity']}
Parking: {row['Parking']}
WiFi: {row['WiFi']}
Payment Methods: {row['Payment Methods']}
Accessibility: {row['Accessibility']}
Latitude: {row['Latitude']}
Longitude: {row['Longitude']}"""
        chunks.append(chunk)
    
    embedder = JinaEmbedder()
    print(f"Embedding {len(chunks)} restaurants...")
    embeddings = await embedder.embed(chunks)
    
    with open("data/vector_store.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "embeddings": embeddings}, f)
    
    print("Vector store created successfully!")

if __name__ == "__main__":
    asyncio.run(ingest_csv())
