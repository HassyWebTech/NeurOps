import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import sys
import os

# Import our settings from config.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    QDRANT_MODE, QDRANT_LOCAL_PATH, QDRANT_COLLECTION,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, REVIEWS_FILE
)


def get_qdrant_client() -> QdrantClient:
    """
    Returns a Qdrant client in local mode.
    Local mode stores data in a folder on disk —
    no server, no Docker, no network connection needed.
    """
    os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
    print(f"Using local Qdrant storage at: {QDRANT_LOCAL_PATH}")
    return QdrantClient(path=QDRANT_LOCAL_PATH)


def load_reviews(filepath: str) -> pd.DataFrame:
    """
    Reads the reviews CSV and prepares it for ingestion.
    We combine the review title and review text into one field
    because together they give more context than either alone.
    """
    print("Loading reviews CSV...")
    df = pd.read_csv(filepath)

    # Drop rows where review message is empty — useless for RAG
    df = df.dropna(subset=["review_comment_message"], how="all")

    # Fill missing titles with empty string so concatenation doesn't break
    df["review_comment_title"] = df["review_comment_title"].fillna("")
    df["review_comment_message"] = df["review_comment_message"].fillna("")

    # Combine title and message into one searchable text field
    df["full_review"] = (
        df["review_comment_title"] + " " + df["review_comment_message"]
    ).str.strip()

    # Remove reviews that ended up empty after cleaning
    df = df[df["full_review"].str.len() > 10]

    print(f"Loaded {len(df)} reviews after cleaning")
    return df


def create_collection(client: QdrantClient):
    """
    Creates a Qdrant collection if it doesn't already exist.
    Running ingest multiple times won't wipe existing data.
    """
    existing = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION in existing:
        print(f"Collection '{QDRANT_COLLECTION}' already exists. Skipping creation.")
        return

    print(f"Creating collection '{QDRANT_COLLECTION}'...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,  # Must match our embedding model (384)
            distance=Distance.COSINE   # Measures similarity by angle, not distance
        )
    )
    print("Collection created.")


def ingest(batch_size: int = 100):
    """
    Main ingestion pipeline.
    Processes reviews in batches to keep memory usage manageable.
    """
    # Step 1: Connect to Qdrant in local mode
    client = get_qdrant_client()

    # Step 2: Load embedding model
    # Downloads once from HuggingFace, cached locally after that
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Step 3: Create collection if needed
    create_collection(client)

    # Step 4: Load and clean reviews
    df = load_reviews(REVIEWS_FILE)

    # Step 5: Process in batches
    total = len(df)
    ingested = 0

    print(f"Starting ingestion of {total} reviews in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        batch = df.iloc[start: start + batch_size]
        texts = batch["full_review"].tolist()

        # Convert texts to vectors locally — no API call needed
        embeddings = model.encode(texts, show_progress_bar=False)

        # Build Qdrant points — vector + metadata
        points = []
        for i, (_, row) in enumerate(batch.iterrows()):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i].tolist(),
                    payload={
                        "review_id": row.get("review_id", ""),
                        "order_id": row.get("order_id", ""),
                        "score": int(row.get("review_score", 0)),
                        "text": row["full_review"],
                    }
                )
            )

        # Upload batch to Qdrant
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )

        ingested += len(batch)
        print(f"Progress: {ingested}/{total} reviews ingested")

    print(f"\nIngestion complete. {ingested} reviews stored in Qdrant.")


if __name__ == "__main__":
    ingest()