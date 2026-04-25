import os
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Document,
    OrderBy,
    OrderByQuery,
    Direction,
    PayloadSchemaType,
)
from menu import ITEMS
from query_router import parse_search_intent

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
    client.create_payload_index(
        collection_name=collection_name,
        field_name="price_cents",
        field_schema=PayloadSchemaType.INTEGER,
    )


def parse_price_to_cents(price: str) -> int:
    match = re.search(r"(\d+)(?:\.(\d{1,2}))?", price)
    if not match:
        raise ValueError(f"Invalid price format: {price}")
    dollars = int(match.group(1))
    cents_part = match.group(2) or "0"
    cents = int(cents_part.ljust(2, "0"))
    return dollars * 100 + cents

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
                "price_cents": parse_price_to_cents(item[2]),
                "category": item[3],
            }
        )
        points.append(point)

    return client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )

def print_results(query: str, points):
    print(f">> Query: {query}")
    for result in points:
        print(f"Item: {result.payload.get('name', 'N/A')}")
        if result.score is not None:
            print(f"Score: {result.score}")
        print(f"Description: {result.payload['description'][:150]}...")
        print(f"Price: {result.payload.get('price', 'N/A')}")
        print("---")


def search(client: QdrantClient, collection_name: str, query: str, limit: int = 10):
    intent = parse_search_intent(query)

    if intent.price_order and not intent.semantic_text:
        direction = Direction.ASC if intent.price_order == "asc" else Direction.DESC
        results = client.query_points(
            collection_name=collection_name,
            query=OrderByQuery(
                order_by=OrderBy(key="price_cents", direction=direction),
            ),
            with_payload=True,
            limit=limit,
        )
        print_results(query, results.points)
        return

    if intent.price_order and intent.semantic_text:
        internal_limit = max(limit * 8, 30)
        results = client.query_points(
            collection_name=collection_name,
            query=Document(
                text=intent.semantic_text,
                model=EMBEDDING_MODEL,
            ),
            with_payload=True,
            limit=internal_limit,
        )
        reverse = intent.price_order == "desc"
        sorted_points = sorted(
            results.points,
            key=lambda point: point.payload.get("price_cents", 0),
            reverse=reverse,
        )[:limit]
        print_results(query, sorted_points)
        return

    results = client.query_points(
        collection_name=collection_name,
        query=Document(
            text=intent.semantic_text or query,
            model=EMBEDDING_MODEL,
        ),
        with_payload=True,
        limit=limit,
    )
    print_results(query, results.points)

if __name__ == "__main__":
    client = get_client()
    create_collection(client, COLLECTION_NAME)
    load_items(client, COLLECTION_NAME)

    search(client, COLLECTION_NAME, "vegetarian dishes", limit=3)
    search(client, COLLECTION_NAME, "the cheapest dish", limit=3)
    search(client, COLLECTION_NAME, "cheapest seafood", limit=3)
