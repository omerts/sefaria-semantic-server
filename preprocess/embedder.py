"""
Generate embeddings for chunks
"""

# Import SSL configuration FIRST - this configures SSL once for all modules
import os
from utils.ssl_config import configure_ssl

configure_ssl()

from typing import List, Optional
import numpy as np


class Embedder:
    """Class for generating embeddings"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Args:
            model_name: Multilingual model name for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 1024  # bge-m3 default
        self._load_model()

    def _load_model(self):
        """Loads the embedding model"""
        try:
            # SSL is already configured by utils.ssl_config when module is imported
            # Set timeout for HuggingFace downloads
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes timeout

            # Check if it's sambert (sentence-transformers) or bge-m3 (FlagEmbedding)
            if self.model_name == "MPA/sambert" or "sambert" in self.model_name.lower():
                # For HuggingFace models, use huggingface_hub directly to download first
                # This is more reliable than letting sentence-transformers handle the download

                # Try to download model using huggingface_hub first, then load with sentence-transformers
                try:
                    from huggingface_hub import snapshot_download
                    import tempfile

                    print(
                        f"Downloading model {self.model_name} using huggingface_hub..."
                    )
                    print(
                        "⚠ This may take several minutes for first-time download (737MB)..."
                    )

                    # Download to HuggingFace cache directory
                    cache_dir = os.path.join(
                        os.path.expanduser("~"), ".cache", "huggingface", "hub"
                    )
                    os.makedirs(cache_dir, exist_ok=True)

                    # Download the model with resume support and increased retries
                    try:
                        os.environ["REQUESTS_CA_BUNDLE"] = (
                            "/Users/netanel/documents-editor/torah-source-finder/ssl_cert.pem"
                        )

                        local_dir = snapshot_download(
                            repo_id=self.model_name,
                            cache_dir=cache_dir,
                            ignore_patterns=[
                                "*.md",
                                "*.txt",
                            ],  # Skip documentation files
                            # force_download=True,  # Resume if partially downloaded
                            max_workers=1,  # Use single worker to avoid connection issues
                        )
                        print(f"✓ Model downloaded to: {local_dir}")
                    except Exception as download_error:
                        # If snapshot_download fails, try downloading individual files
                        error_str = str(download_error).lower()
                        if (
                            "retries" in error_str
                            or "timeout" in error_str
                            or "connection" in error_str
                        ):
                            print(
                                f"⚠ Snapshot download failed (likely connection issue): {download_error}"
                            )
                            print(
                                "⚠ The model file is large (737MB) and may need multiple attempts."
                            )
                            print(
                                "💡 Tip: Try running the pipeline again - it will resume from where it stopped."
                            )
                            print(
                                "💡 Or download manually: python -c \"from huggingface_hub import snapshot_download; snapshot_download('MPA/sambert')\""
                            )
                            raise
                        else:
                            raise

                    # Now load with sentence-transformers from local directory
                    from sentence_transformers import SentenceTransformer

                    print(f"Loading model from local cache...")
                    self.model = SentenceTransformer(local_dir, device="cpu")
                    self.embedding_dim = 768  # sambert dimension
                    print("✓ Model loaded")

                except ImportError:
                    # If huggingface_hub not available, fall back to direct loading
                    print("⚠ huggingface_hub not available, trying direct loading...")
                    from sentence_transformers import SentenceTransformer

                    print(f"Loading Hebrew embedding model: {self.model_name}...")
                    print(
                        "⚠ This may take several minutes for first-time download (737MB)..."
                    )
                    self.model = SentenceTransformer(
                        self.model_name,
                        trust_remote_code=True,
                        device="cpu",
                    )
                    self.embedding_dim = 768
                    print("✓ Model loaded")

                except Exception as e:
                    # If download fails, try direct loading as fallback
                    error_str = str(e).lower()
                    print(f"⚠ Download attempt failed: {e}")
                    print("⚠ Trying direct loading with sentence-transformers...")

                    from sentence_transformers import SentenceTransformer

                    try:
                        self.model = SentenceTransformer(
                            self.model_name, trust_remote_code=True, device="cpu"
                        )
                        self.embedding_dim = 768
                        print("✓ Model loaded directly")
                    except Exception as e2:
                        print(f"⚠ Direct loading also failed: {e2}")
                        print(
                            "💡 Tip: Try downloading the model manually or check your internet connection"
                        )
                        raise
            else:
                # Default: bge-m3
                from FlagEmbedding import BGEM3FlagModel

                print(f"Loading embedding model: {self.model_name}...")
                self.model = BGEM3FlagModel(self.model_name, use_fp16=False)
                self.embedding_dim = 1024  # bge-m3 default
                print("✓ Model loaded")
        except ImportError as e:
            if "sentence_transformers" in str(e):
                print(
                    "⚠ sentence-transformers not installed. Install: pip install sentence-transformers"
                )
            else:
                print(
                    "⚠ FlagEmbedding not installed. Install: pip install FlagEmbedding"
                )
            self.model = None
        except Exception as e:
            error_str = str(e).lower()
            print(f"⚠ Error loading model: {e}")
            # If certificate failed or SSL error, try disabling SSL verification
            if (
                "certificate" in error_str
                or "ssl" in error_str
                or "verify" in error_str
                or "tls" in error_str
            ):
                print(
                    "⚠ SSL/TLS verification failed. SSL should already be configured."
                )
                print("💡 Tip: Set DISABLE_SSL_VERIFY=1 to disable SSL verification")
                self.model = None
            else:
                print("💡 Tip: Set DISABLE_SSL_VERIFY=1 to disable SSL verification")
                self.model = None

    def embed(self, text: str, normalize: bool = True) -> Optional[List[float]]:
        """
        Creates an embedding for text

        Args:
            text: Text to embed
            normalize: Whether to normalize the vector

        Returns:
            List of floats (embedding vector) or None
        """
        if not self.model or not text:
            return None

        try:
            # Check if it's sentence-transformers or FlagEmbedding
            from sentence_transformers import SentenceTransformer

            is_sentence_transformer = isinstance(self.model, SentenceTransformer)
        except ImportError:
            is_sentence_transformer = False

        try:
            if is_sentence_transformer:
                # sentence-transformers API
                embedding = self.model.encode(text, normalize_embeddings=normalize)
                if hasattr(embedding, "tolist"):
                    return embedding.tolist()
                elif isinstance(embedding, list):
                    return embedding
                else:
                    return list(embedding)
            else:
                # bge-m3 FlagEmbedding API
                # BGEM3FlagModel.encode() returns a dict with 'dense_vecs', 'lexical_weights', 'colbert_vecs'
                result = self.model.encode(
                    text,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )

                # Extract dense vectors from the result
                if isinstance(result, dict) and "dense_vecs" in result:
                    embedding = result["dense_vecs"]
                else:
                    embedding = result

                if isinstance(embedding, np.ndarray):
                    # Handle 2D array (batch of 1)
                    if len(embedding.shape) == 2:
                        embedding = embedding[0]
                    return embedding.tolist()
                elif isinstance(embedding, list):
                    return embedding
                else:
                    return list(embedding)

        except Exception as e:
            print(f"Error creating embedding: {e}")
            return None

    def embed_batch(
        self, texts: List[str], batch_size: int = 32, normalize: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Creates embeddings for a list of texts

        Args:
            texts: List of texts
            batch_size: Batch size for processing
            normalize: Whether to normalize

        Returns:
            List of embeddings
        """
        if not self.model:
            return [None] * len(texts)

        # Check if it's sentence-transformers or FlagEmbedding
        try:
            from sentence_transformers import SentenceTransformer

            is_sentence_transformer = isinstance(self.model, SentenceTransformer)
        except ImportError:
            is_sentence_transformer = False

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                if is_sentence_transformer:
                    # sentence-transformers API
                    batch_embeddings = self.model.encode(
                        batch, normalize_embeddings=normalize, batch_size=len(batch)
                    )
                    if hasattr(batch_embeddings, "tolist"):
                        batch_embeddings = batch_embeddings.tolist()
                    results.extend(batch_embeddings)
                else:
                    # BGEM3FlagModel.encode() returns a dict with 'dense_vecs', 'lexical_weights', 'colbert_vecs'
                    batch_result = self.model.encode(
                        batch,
                        batch_size=len(batch),
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False,
                    )

                    # Extract dense vectors from the result
                    if isinstance(batch_result, dict) and "dense_vecs" in batch_result:
                        batch_embeddings = batch_result["dense_vecs"]
                    else:
                        batch_embeddings = batch_result

                    if isinstance(batch_embeddings, np.ndarray):
                        # Convert to list of lists
                        batch_embeddings = batch_embeddings.tolist()

                    # Ensure we have a list of embeddings
                    if isinstance(batch_embeddings, list) and len(batch_embeddings) > 0:
                        # Check if first element is a list (list of vectors) or number (single vector)
                        if isinstance(batch_embeddings[0], (list, np.ndarray)):
                            results.extend(batch_embeddings)
                        else:
                            # Single embedding for entire batch - shouldn't happen but handle it
                            results.append(batch_embeddings)
                    else:
                        results.extend([None] * len(batch))

            except Exception as e:
                print(f"Error in batch embedding: {e}")
                results.extend([None] * len(batch))

        return results


# Global instance
_embedder_instance = None
_embedder_model_name = None


def get_embedder(model_name: str = None) -> Embedder:
    """Returns embedder instance (singleton or new instance if model_name differs)"""
    global _embedder_instance, _embedder_model_name

    # If no model_name provided, use default
    if model_name is None:
        model_name = "BAAI/bge-m3"

    # If instance doesn't exist or model changed, create new instance
    if _embedder_instance is None or _embedder_model_name != model_name:
        _embedder_instance = Embedder(model_name=model_name)
        _embedder_model_name = model_name

    return _embedder_instance


def embed(text: str) -> Optional[List[float]]:
    """Convenience function for creating embedding"""
    embedder = get_embedder()
    return embedder.embed(text)


def process_chunks_with_embeddings(
    chunks_file: str = "data/chunks/all_chunks.json",
    output_file: str = "data/chunks/chunks_with_embeddings.json",
    books: List[str] = None,
    model_name: str = None,
):
    """Processes chunks and adds embeddings"""
    import json
    from pathlib import Path
    from typing import List
    import os

    # Get model name from parameter, env var, or default
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

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
    print(f"Using embedding model: {model_name}")

    embedder = get_embedder(model_name)
    if not embedder.model:
        print("⚠ Cannot create embeddings - model not available")
        return

    print("Creating embeddings...")
    texts = [chunk.get("text", "") for chunk in chunks]
    embeddings = embedder.embed_batch(texts, batch_size=16)

    # Add embeddings and model name to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        chunk["embedding_model"] = model_name  # Add embedding model name

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    abs_path = output_path.resolve()
    file_size = output_path.stat().st_size
    successful_embeddings = sum(1 for e in embeddings if e is not None)
    print(f"\n{'='*60}")
    print(f"Embedding Summary:")
    print(f"  Saved to: {abs_path}")
    print(f"  File size: {file_size} bytes")
    print(f"  Embeddings created: {successful_embeddings}/{len(embeddings)}")
    print(f"  Model used: {model_name}")
    print(f"{'='*60}")
    print(f"  Saved to: {abs_path}")
    print(f"  File size: {file_size} bytes")
    print(f"  Embeddings created: {successful_embeddings}/{len(embeddings)}")
    print(f"  Model used: {model_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_chunks_with_embeddings()
