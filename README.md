# Qdrant vector database demo

Small demo that stores synthetic menu items in [Qdrant](https://qdrant.tech/), embeds them with **Qdrant Cloud Inference** using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional cosine vectors), and runs a sample semantic search.

## What it does

1. Connects to Qdrant using `QDRANT_URL` and `QDRANT_API_KEY`.
2. Recreates a collection named `items` (drops it if it already exists).
3. Upserts points from `menu.py` (`ITEMS`: name, description, price, category). Vectors are produced from the concatenated name and description via cloud inference.
4. Prints top results for the query `"vegetarian dishes"` (see `main.py`).

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
