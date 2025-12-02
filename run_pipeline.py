#!/usr/bin/env python3
"""
Script to run the full pipeline
"""
import sys
from pathlib import Path


def run_step(
    step_name: str, module_path: str, description: str, function_name: str = None
):
    """Runs a pipeline step"""
    print(f"\n{'='*60}")
    print(f"Step: {step_name}")
    print(f"Description: {description}")
    print(f"{'='*60}\n")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # If function_name is specified, call it
        if function_name and hasattr(module, function_name):
            func = getattr(module, function_name)
            func()

        print(f"✓ {step_name} completed\n")
        return True
    except Exception as e:
        print(f"✗ Error in {step_name}: {e}\n")
        return False


def main():
    """Runs the full pipeline"""
    base_dir = Path(__file__).parent

    steps = [
        (
            "1. Download data",
            str(base_dir / "ingestion" / "download_sefaria.py"),
            "Downloads books from Sefaria API",
            "run_download",
        ),  # Call run_download function
        (
            "2. Normalize data",
            str(base_dir / "ingestion" / "normalize.py"),
            "Normalizes and cleans raw data",
            "normalize_all",
        ),  # Call normalize_all function
        (
            "3. Create chunks",
            str(base_dir / "preprocess" / "chunker.py"),
            "Divides text into chunks",
            "process_all_entries",
        ),  # Call process_all_entries function
        (
            "4. Create embeddings",
            str(base_dir / "preprocess" / "embedder.py"),
            "Creates embeddings for chunks",
            "process_chunks_with_embeddings",
        ),  # Call process_chunks_with_embeddings function
        (
            "5. Index in Qdrant",
            str(base_dir / "preprocess" / "indexer.py"),
            "Uploads chunks to Qdrant",
            "index_chunks",
        ),  # Call index_chunks function
    ]

    print("=" * 60)
    print("Torah Source Finder - Pipeline")
    print("=" * 60)

    for step_info in steps:
        if len(step_info) == 4:
            step_name, module_path, description, function_name = step_info
        else:
            step_name, module_path, description = step_info
            function_name = None

        if not run_step(step_name, module_path, description, function_name):
            response = input("Continue to next step? (y/n): ").strip().lower()
            if response != "y":
                print("Pipeline stopped")
                sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Pipeline completed successfully!")
    print("=" * 60)
    print("\nTo run the API:")
    print("  uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
