"""
Index chunks in Qdrant
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from typing import List, Dict, Any
import json
from pathlib import Path


class QdrantIndexer:
    """Class for indexing in Qdrant"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "sefaria_chunks",
    ):
        """
        Args:
            host: Qdrant address
            port: Qdrant port
            collection_name: Collection name
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dim = 1024  # bge-m3 default

    def ensure_collection(self, vector_size: int = 1024, distance: str = "Cosine"):
        """
        Creates collection if it doesn't exist

        Args:
            vector_size: Vector size
            distance: Distance type (Cosine, Dot, Euclidean)
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name in collection_names:
                # Check if vector size matches
                existing_collection = self.client.get_collection(self.collection_name)
                existing_vector_size = (
                    existing_collection.config.params.vectors.size
                    if hasattr(existing_collection.config.params, "vectors")
                    else None
                )
                
                if existing_vector_size == vector_size:
                    print(f"Collection '{self.collection_name}' already exists with correct vector size ({vector_size})")
                    return
                else:
                    print(f"Collection '{self.collection_name}' exists but with different vector size:")
                    print(f"  Existing: {existing_vector_size}, Required: {vector_size}")
                    print(f"  Deleting existing collection and creating new one...")
                    self.client.delete_collection(self.collection_name)
                    print(f"  ✓ Deleted old collection")

            distance_map = {
                "Cosine": qm.Distance.COSINE,
                "Dot": qm.Distance.DOT,
                "Euclidean": qm.Distance.EUCLID,
            }

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, qm.Distance.COSINE),
                ),
            )
            print(f"✓ Collection '{self.collection_name}' created with vector size {vector_size}")

        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Uploads chunks to Qdrant

        Args:
            chunks: List of chunks with embeddings
            batch_size: Batch size for upload
        """
        if not chunks:
            print("No chunks to upload")
            return

        points = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if not embedding:
                print(f"⚠ Chunk {chunk.get('chunk_id')} without embedding - skipping")
                continue

            payload = {
                "chunk_id": chunk.get("chunk_id"),
                "parent_id": chunk.get("parent_id"),
                "sefaria_ref": chunk.get("sefaria_ref"),
                "book": chunk.get("book"),
                "category": chunk.get("category", []),
                "text": chunk.get("text"),
                "position": chunk.get("position", 0),
                "chunk_type": chunk.get("chunk_type", "default"),
                "embedding_model": chunk.get(
                    "embedding_model", "BAAI/bge-m3"
                ),  # Add embedding model name
            }

            # Add only fields that exist in chunk (no null values)
            # Fields are added dynamically based on addressTypes
            for field in [
                "masechet",
                "daf",
                "amud",
                "chapter",
                "verse",
                "perek",
                "mishnah",
                "part",
                "siman",
                "seif",
            ]:
                if field in chunk and chunk[field] is not None:
                    payload[field] = chunk[field]

            # Add summary if exists
            if "summary" in chunk:
                payload["summary"] = chunk["summary"]

            points.append(
                qm.PointStruct(
                    id=hash(chunk.get("chunk_id", ""))
                    % (2**63),  # Qdrant requires int64
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upload in batches
        print(f"Uploading {len(points)} points to Qdrant...")
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                print(
                    f"✓ Uploaded {min(i + batch_size, len(points))}/{len(points)} points"
                )
            except Exception as e:
                print(f"✗ Error uploading batch {i//batch_size + 1}: {e}")

    def search(
        self, query_vector: List[float], limit: int = 10, score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search in Qdrant

        Args:
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum score threshold

        Returns:
            List of results
        """
        try:
            # Use query_points() method for newer qdrant-client versions
            # query can be a list[float] directly
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,  # Can be list[float] directly
                limit=limit,
                score_threshold=score_threshold,
            )

            return [
                {"score": hit.score, "payload": hit.payload} for hit in results.points
            ]
        except Exception as e:
            # Fallback to older API if query_points doesn't work
            try:
                # Try with search() for older versions
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                )

                return [{"score": hit.score, "payload": hit.payload} for hit in results]
            except Exception as e2:
                print(f"Error in search: {e2}")
                return []

    def get_collection_info(self) -> Dict:
        """Returns collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            # Handle different Qdrant API versions
            points_count = getattr(info, "points_count", None)
            if points_count is None:
                # Try to get from vectors_count or other attributes
                points_count = getattr(info, "vectors_count", 0)

            return {
                "points_count": points_count or 0,
                "vectors_count": getattr(info, "vectors_count", points_count or 0),
                "config": {
                    "vector_size": (
                        info.config.params.vectors.size
                        if hasattr(info.config.params, "vectors")
                        else 0
                    ),
                    "distance": (
                        str(info.config.params.vectors.distance)
                        if hasattr(info.config.params, "vectors")
                        else "Cosine"
                    ),
                },
            }
        except Exception as e:
            print(f"Error getting info: {e}")
            return {"points_count": 0, "vectors_count": 0}

    def delete_by_book(self, book: str) -> int:
        """
        Deletes all points for a specific book

        Args:
            book: Book name to delete

        Returns:
            Number of points deleted
        """
        try:
            # Create filter for the book
            filter_condition = qm.Filter(
                must=[qm.FieldCondition(key="book", match=qm.MatchValue(value=book))]
            )

            # Count points before deletion
            count = 0
            offset = None
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=1000,
                    offset=offset,
                )
                if not scroll_result[0]:
                    break
                count += len(scroll_result[0])
                offset = scroll_result[1]
                if not offset:
                    break

            # Delete points matching the filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qm.FilterSelector(filter=filter_condition),
            )

            return count
        except Exception as e:
            print(f"Error deleting by book: {e}")
            raise

    def delete_all(self) -> int:
        """
        Deletes all points from the collection

        Returns:
            Number of points deleted
        """
        try:
            # Get count before deletion
            info = self.get_collection_info()
            count = info.get("points_count", 0)

            # Delete all points - scroll through all and delete by IDs
            # More efficient: use scroll to get all IDs, then delete
            all_ids = []
            offset = None
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name, limit=1000, offset=offset
                )
                if not scroll_result[0]:
                    break
                all_ids.extend([point.id for point in scroll_result[0]])
                offset = scroll_result[1]
                if not offset:
                    break

            if all_ids:
                # Delete in batches
                batch_size = 1000
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i : i + batch_size]
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=qm.PointIdsList(points=batch_ids),
                    )

            return len(all_ids) if all_ids else count
        except Exception as e:
            print(f"Error deleting all: {e}")
            raise


def _get_collection_name_from_model(embedding_model: str = None) -> str:
    """
    Generates collection name from embedding model name
    
    Args:
        embedding_model: Model name (e.g., "MPA/sambert", "BAAI/bge-m3")
    
    Returns:
        Collection name (e.g., "sefaria_chunks_MPA_sambert", "sefaria_chunks_BAAI_bge-m3")
    """
    if embedding_model:
        # Normalize model name: replace "/" and "-" with "_"
        model_suffix = embedding_model.replace("/", "_").replace("-", "_")
        return f"sefaria_chunks_{model_suffix}"
    return "sefaria_chunks"


def index_chunks(
    chunks_file: str = "data/chunks/chunks_with_embeddings.json",
    host: str = "localhost",
    port: int = 6333,
    books: List[str] = None,
    embedding_model: str = None,
):
    """Convenience function for indexing chunks"""
    from typing import List

    chunks_path = Path(chunks_file).resolve()
    if not chunks_path.exists():
        print(f"File not found: {chunks_path}")
        return

    print(f"\nLoading chunks from: {chunks_path}")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    # Filter by books if provided
    if books:
        # Normalize book names for comparison
        books_normalized = {book.replace("/", "_") for book in books}
        original_count = len(chunks)
        chunks = [
            chunk
            for chunk in chunks
            if chunk.get("book", "").replace("/", "_") in books_normalized
        ]
        print(
            f"Filtered from {original_count} to {len(chunks)} chunks for specified books"
        )

    print(f"Found {len(chunks)} chunks")

    # Check embedding size and get embedding model from chunks if not provided
    sample_embedding = None
    chunk_embedding_model = None
    for chunk in chunks:
        if chunk.get("embedding"):
            sample_embedding = chunk["embedding"]
            chunk_embedding_model = chunk.get("embedding_model")
            break

    if not sample_embedding:
        print("⚠ No embeddings found in chunks")
        return

    # Use embedding_model from parameter, or from chunk, or default
    if not embedding_model:
        embedding_model = chunk_embedding_model or "BAAI/bge-m3"

    vector_size = len(sample_embedding)
    collection_name = _get_collection_name_from_model(embedding_model)
    
    print(f"Vector size: {vector_size}")
    print(f"Embedding model: {embedding_model}")
    print(f"Qdrant server: {host}:{port}")
    print(f"Collection name: {collection_name}\n")

    # Create indexer with collection name based on embedding model
    indexer = QdrantIndexer(host=host, port=port, collection_name=collection_name)
    indexer.embedding_dim = vector_size

    # Create collection
    print("Creating collection...")
    indexer.ensure_collection(vector_size=vector_size)

    # Upload chunks
    print("Uploading chunks...")
    indexer.upsert_chunks(chunks)

    # Final info
    info = indexer.get_collection_info()
    print(f"\n{'='*60}")
    print(f"Indexing Summary:")
    print(f"  Qdrant server: {host}:{port}")
    print(f"  Collection: {collection_name}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Points indexed: {info.get('points_count', 0)}")
    print(f"  Vectors indexed: {info.get('vectors_count', 0)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys

    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 6333

    index_chunks(host=host, port=port)
