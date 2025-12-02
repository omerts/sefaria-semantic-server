"""
API endpoints for pipeline and deletion operations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import sys
import importlib.util
from .dependencies import indexer

router = APIRouter()


class PipelineRequest(BaseModel):
    """Pipeline run request"""

    books: Optional[List[str]] = None  # If None, processes all books
    embedding_model: Optional[str] = (
        None  # Embedding model name (e.g., "MPA/sambert", "BAAI/bge-m3")
    )
    language: Optional[str] = (
        "hebrew"  # Language/version for Sefaria API (e.g., "hebrew", "english")
    )


class PipelineResponse(BaseModel):
    """Pipeline run response"""

    message: str
    books: Optional[List[str]] = None
    status: str  # "started" or "completed"


class DeleteRequest(BaseModel):
    """Delete request"""

    book: Optional[str] = None  # If None, deletes all
    embedding_model: Optional[str] = (
        None  # Embedding model name (e.g., "MPA/sambert", "BAAI/bge-m3")
    )


class DeleteResponse(BaseModel):
    """Delete response"""

    message: str
    deleted_count: int
    book: Optional[str] = None


def run_pipeline_step(
    step_name: str,
    module_path: str,
    description: str,
    function_name: str = None,
    books: List[str] = None,
    embedding_model: str = None,
    language: str = "hebrew",
):
    """Runs a pipeline step"""
    print(f"\n{'='*60}")
    print(f"Step: {step_name}")
    print(f"Description: {description}")
    if books:
        print(f"Books: {', '.join(books)}")
    if embedding_model:
        print(f"Embedding model: {embedding_model}")
    if language:
        print(f"Language: {language}")
    print(f"{'='*60}\n")

    try:
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # If function_name is specified, call it
        if function_name and hasattr(module, function_name):
            func = getattr(module, function_name)

            # Check function parameters
            import inspect

            sig = inspect.signature(func)
            params = {}

            if "books" in sig.parameters:
                params["books"] = books
            if "model_name" in sig.parameters and embedding_model:
                params["model_name"] = embedding_model
            if "embedding_model" in sig.parameters and embedding_model:
                params["embedding_model"] = embedding_model
            if "version" in sig.parameters and language:
                params["version"] = language

            if params:
                func(**params)
            else:
                func()

        print(f"✓ {step_name} completed\n")
        return True
    except Exception as e:
        print(f"✗ Error in {step_name}: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def run_pipeline_async(
    books: List[str] = None, embedding_model: str = None, language: str = "hebrew"
):
    """Runs the full pipeline asynchronously"""
    base_dir = Path(__file__).parent.parent

    steps = [
        (
            "1. Download data",
            str(base_dir / "ingestion" / "download_sefaria.py"),
            "Downloads books from Sefaria API",
            "run_download",
        ),
        (
            "2. Normalize data",
            str(base_dir / "ingestion" / "normalize.py"),
            "Normalizes and cleans raw data",
            "normalize_all",
        ),
        (
            "3. Create chunks",
            str(base_dir / "preprocess" / "chunker.py"),
            "Divides text into chunks",
            "process_all_entries",
        ),
        (
            "4. Create embeddings",
            str(base_dir / "preprocess" / "embedder.py"),
            "Creates embeddings for chunks",
            "process_chunks_with_embeddings",
        ),
        (
            "5. Index in Qdrant",
            str(base_dir / "preprocess" / "indexer.py"),
            "Uploads chunks to Qdrant",
            "index_chunks",
        ),
    ]

    print("=" * 60)
    print("Torah Source Finder - Pipeline")
    if books:
        print(f"Processing books: {', '.join(books)}")
    if embedding_model:
        print(f"Embedding model: {embedding_model}")
    if language:
        print(f"Language: {language}")
    print("=" * 60)

    for step_info in steps:
        if len(step_info) == 4:
            step_name, module_path, description, function_name = step_info
        else:
            step_name, module_path, description = step_info
            function_name = None

        if not run_pipeline_step(
            step_name,
            module_path,
            description,
            function_name,
            books,
            embedding_model,
            language,
        ):
            print(f"Pipeline stopped at step: {step_name}")
            return False

    print("\n" + "=" * 60)
    print("✓ Pipeline completed successfully!")
    print("=" * 60)
    return True


@router.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Run the full pipeline

    If books list is provided, only those books will be processed.
    If books is None or empty, all books will be processed.
    If embedding_model is provided, it will be used for embeddings (e.g., "MPA/sambert", "BAAI/bge-m3").
    If embedding_model is None, defaults to "BAAI/bge-m3".
    If language is provided, it will be used for Sefaria API (e.g., "hebrew", "english").
    If language is None, defaults to "hebrew".
    """
    books = request.books if request.books else None
    embedding_model = request.embedding_model
    language = request.language if request.language else "hebrew"

    # Add pipeline to background tasks
    background_tasks.add_task(
        run_pipeline_async,
        books=books,
        embedding_model=embedding_model,
        language=language,
    )

    return PipelineResponse(
        message="Pipeline started in background", books=books, status="started"
    )


@router.delete("/delete", response_model=DeleteResponse)
async def delete_from_vector_db(
    book: Optional[str] = None, embedding_model: Optional[str] = None
):
    """
    Delete from vector database

    - If book query parameter is provided, deletes all chunks for that book
    - If book is not provided, deletes all chunks from the database
    - Uses collection based on embedding_model query parameter
    """
    from preprocess.indexer import QdrantIndexer, _get_collection_name_from_model
    import os

    try:
        # Get embedding model from query parameter or use default
        embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL_NAME", "BAAI/bge-m3"
        )

        # Create indexer with collection name based on embedding model
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        collection_name = _get_collection_name_from_model(embedding_model)
        indexer = QdrantIndexer(
            host=qdrant_host, port=qdrant_port, collection_name=collection_name
        )

        if book:
            # Delete by book
            deleted_count = indexer.delete_by_book(book)
            return DeleteResponse(
                message=f"Deleted {deleted_count} chunks for book '{book}' from collection '{collection_name}'",
                deleted_count=deleted_count,
                book=book,
            )
        else:
            # Delete all
            deleted_count = indexer.delete_all()
            return DeleteResponse(
                message=f"Deleted all {deleted_count} chunks from collection '{collection_name}'",
                deleted_count=deleted_count,
                book=None,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
