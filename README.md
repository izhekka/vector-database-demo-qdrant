# Qdrant vector database demo

Small demo that stores synthetic menu items in [Qdrant](https://qdrant.tech/), embeds them with **Qdrant Cloud Inference** using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional cosine vectors), and runs a sample semantic search.

## What it does

1. Connects to Qdrant using `QDRANT_URL` and `QDRANT_API_KEY`.
2. Recreates a collection named `items` (drops it if it already exists).
3. Upserts points from `menu.py` (`ITEMS`: name, description, price, category). Vectors are produced from the concatenated name and description via cloud inference.
4. Supports three query modes:
   - semantic search (for example: `"vegetarian dishes"`)
   - price-aware sort (for example: `"the cheapest dish"`)
   - hybrid semantic + price intent (for example: `"cheapest seafood"`)

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed.
- A Qdrant instance with **cloud inference** enabled (e.g. [Qdrant Cloud](https://cloud.qdrant.io/)) so `Document(text=..., model=...)` works. Use the cluster URL and API key from the cloud console.

## Setup

From the project root:

```bash
uv venv
uv pip install -r requirements.txt
```

Copy environment template and fill in your cluster credentials:

```bash
cp .env.example .env
```

Edit `.env`:

- `QDRANT_URL` — REST URL of your Qdrant cluster (e.g. `https://xxxx.cloud.qdrant.io`).
- `QDRANT_API_KEY` — API key for that cluster.

## Run

With the virtual environment created as above, `uv run` uses the project `.venv` automatically:

```bash
uv run main.py
```

Equivalent:

```bash
uv run python main.py
```

## Price-aware and hybrid behavior

The app stores two price payloads per point:

- `price` as the original string (for output), e.g. `"$13.95"`
- `price_cents` as an integer (for sorting/filtering), e.g. `1395`

`price_cents` is indexed in Qdrant with `create_payload_index(..., field_schema=INTEGER)`.

At query time:

- **Semantic-only** uses vector search with `Document`.
- **Price-only** queries (like `"cheapest"` / `"most expensive"`) use `OrderByQuery` on `price_cents`.
- **Hybrid** queries (like `"cheapest seafood"`) first run vector search on cleaned semantic text (`"seafood"`), then rerank returned candidates by `price_cents`.
