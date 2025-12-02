"""
Dependencies for FastAPI - singleton instances
"""

from models.reranker import ReRanker
from qdrant_client import QdrantClient
from preprocess.indexer import QdrantIndexer, _get_collection_name_from_model
import os


# Re-Ranker instance
reranker = ReRanker(model_path=os.getenv("RERANKER_MODEL_PATH", None))

# Qdrant client
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

# Qdrant indexer - use collection name based on embedding model
# Get embedding model from env var or default to BAAI/bge-m3
embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
qdrant_collection = _get_collection_name_from_model(embedding_model)
indexer = QdrantIndexer(
    host=qdrant_host, port=qdrant_port, collection_name=qdrant_collection
)
