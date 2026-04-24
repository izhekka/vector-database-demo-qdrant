import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Document,
)
from menu import ITEMS

load_dotenv()

COLLECTION_NAME = "items"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        cloud_inference=True,
    )

def create_collection(client: QdrantClient, collection_name: str):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

def load_items(client: QdrantClient, collection_name: str):
    points = []
    for index, item in enumerate(ITEMS):
        point = PointStruct(
            id=index,
            vector=Document(
                text=f"{item[0]} {item[1]}",
                model=EMBEDDING_MODEL,
            ),
            payload={
                "name": item[0],
                "description": item[1],
                "price": item[2],
                "category": item[3],
            }
        )
        points.append(point)

    return client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )

def search(client: QdrantClient, collection_name: str, query: str, limit: int = 10):
    results = client.query_points(
        collection_name=collection_name,
        query=Document(
            text=query,
            model=EMBEDDING_MODEL,
        ),
        with_payload=True,
        limit=limit,
    )

    print(f">> Query: {query}")
    for result in results.points:
        print(f"Item: {result.payload.get('name', 'N/A')}")
        print(f"Score: {result.score}")
        print(f"Description: {result.payload['description'][:150]}...")
        print(f"Price: {result.payload.get('price', 'N/A')}")
        print("---")

if __name__ == "__main__":
    client = get_client()
    create_collection(client, COLLECTION_NAME)
    load_items(client, COLLECTION_NAME)

    search(client, COLLECTION_NAME, "vegetarian dishes", limit=5)
