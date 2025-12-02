"""
API endpoints for search
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .dependencies import reranker
from preprocess.indexer import QdrantIndexer, _get_collection_name_from_model
from preprocess.embedder import get_embedder
import os


router = APIRouter()


class SearchRequest(BaseModel):
    """Search request"""

    query: str
    limit: int = 10
    score_threshold: float = 0.0
    book: Optional[str] = None
    category: Optional[List[str]] = None
    embedding_model: Optional[str] = (
        None  # Embedding model name (e.g., "MPA/sambert", "BAAI/bge-m3")
    )


class SearchResult(BaseModel):
    """Search result"""

    sefaria_ref: str
    book: str
    category: List[str]
    text: str
    score: float
    position: int
    chunk_type: str
    # Talmud fields
    masechet: Optional[str] = None
    daf: Optional[int] = None
    amud: Optional[str] = None
    # Tanakh fields
    chapter: Optional[int] = None
    verse: Optional[int] = None
    # Mishnah fields
    perek: Optional[int] = None
    mishnah: Optional[int] = None
    # Shulchan Arukh fields
    part: Optional[str] = None
    siman: Optional[int] = None
    seif: Optional[int] = None

    embedding_model: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response"""

    results: List[SearchResult]
    total: int
    query: str


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for Torah sources

    Flow:
    1. Create embedding for query
    2. Search in Qdrant (using collection based on embedding_model)
    3. Re-rank results
    4. Return sorted results
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Get embedding model from request or use default
    embedding_model = request.embedding_model or os.getenv(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-m3"
    )

    # Create indexer with collection name based on embedding model
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = _get_collection_name_from_model(embedding_model)
    indexer = QdrantIndexer(
        host=qdrant_host, port=qdrant_port, collection_name=collection_name
    )

    # 1. Create embedding for query using the same model as the collection
    embedder = get_embedder(embedding_model)
    if not embedder.model:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load embedding model: {embedding_model}",
        )

    query_embedding = embedder.embed(request.query)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to create query embedding")

    # 2. Search in Qdrant
    # Search with higher limit so re-ranker can choose
    search_limit = min(request.limit * 3, 50)  # Retrieve 3x for better results

    try:
        qdrant_results = indexer.search(
            query_vector=query_embedding,
            limit=search_limit,
            score_threshold=request.score_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    if not qdrant_results:
        return SearchResponse(results=[], total=0, query=request.query)

    # 3. Prepare texts for re-ranking
    texts_for_rerank = []
    payloads = []

    for result in qdrant_results:
        payload = result["payload"]

        # Filter by book if specified
        if request.book and payload.get("book") != request.book:
            continue

        # Filter by category if specified
        if request.category:
            book_categories = payload.get("category", [])
            if not any(cat in book_categories for cat in request.category):
                continue

        # Use text or summary if available
        text = payload.get("text", "")
        if not text and payload.get("summary"):
            text = payload.get("summary")

        texts_for_rerank.append(text)
        payloads.append(payload)

    if not texts_for_rerank:
        return SearchResponse(results=[], total=0, query=request.query)

    # 4. Re-ranking
    rerank_scores = reranker.score(request.query, texts_for_rerank)

    # 5. Sort by re-ranker score
    ranked_results = sorted(
        zip(payloads, texts_for_rerank, rerank_scores),
        key=lambda x: x[2],
        reverse=True,
    )

    # 6. Return results
    results = []
    for payload, text, score in ranked_results[: request.limit]:
        results.append(
            SearchResult(
                sefaria_ref=payload.get("sefaria_ref", ""),
                book=payload.get("book", ""),
                category=payload.get("category", []),
                text=text,
                score=float(score),
                position=payload.get("position", 0),
                chunk_type=payload.get("chunk_type", "default"),
                # Talmud fields
                masechet=payload.get("masechet"),
                daf=payload.get("daf"),
                amud=payload.get("amud"),
                # Tanakh fields
                chapter=payload.get("chapter"),
                verse=payload.get("verse"),
                # Mishnah fields
                perek=payload.get("perek"),
                mishnah=payload.get("mishnah"),
                # Shulchan Arukh fields
                part=payload.get("part"),
                siman=payload.get("siman"),
                seif=payload.get("seif"),
                embedding_model=payload.get("embedding_model"),
            )
        )

    return SearchResponse(results=results, total=len(results), query=request.query)


@router.get("/health")
async def health():
    """System health check"""
    try:
        # Use default collection for health check
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        default_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        collection_name = _get_collection_name_from_model(default_model)
        indexer = QdrantIndexer(
            host=qdrant_host, port=qdrant_port, collection_name=collection_name
        )
        info = indexer.get_collection_info()
        # Get default embedder for health check
        default_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        embedder = get_embedder(default_model)
        return {
            "status": "healthy",
            "embedder": embedder.model is not None,
            "embedding_model": default_model,
            "reranker": reranker.model is not None,
            "qdrant": {
                "connected": True,
                "collection": collection_name,
                "points": info.get("points_count", 0),
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
