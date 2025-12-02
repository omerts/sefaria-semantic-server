"""
Train Re-Ranker Model
"""

import json
from pathlib import Path
from sentence_transformers import CrossEncoder
from typing import List, Tuple


def load_pairs(pairs_file: str = "training/pairs.json") -> List[Tuple[List[str], int]]:
    """
    Loads training pairs for re-ranker

    Args:
        pairs_file: Pairs file

    Returns:
        List of tuples: ([idea, source], label)
    """
    pairs_path = Path(pairs_file)
    if not pairs_path.exists():
        raise FileNotFoundError(f"File not found: {pairs_path}")

    data = json.loads(pairs_path.read_text(encoding="utf-8"))

    examples = []
    for pair in data:
        label = pair.get("label", 0)
        examples.append(([pair["idea"], pair["source"]], label))

    return examples


def train_reranker(
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    pairs_file: str = "training/pairs.json",
    output_dir: str = "models/reranker",
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """
    Trains re-ranker model

    Args:
        base_model: Base model
        pairs_file: Training pairs file
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    print(f"Loading base model: {base_model}")
    model = CrossEncoder(base_model, num_labels=1)

    print("Loading training pairs...")
    train_samples = load_pairs(pairs_file)
    print(f"Found {len(train_samples)} training examples")

    positive_count = sum(1 for _, label in train_samples if label == 1)
    negative_count = sum(1 for _, label in train_samples if label == 0)
    print(f"  Positive: {positive_count}")
    print(f"  Negative: {negative_count}")

    if not train_samples:
        print("⚠ No training examples!")
        return

    # Training
    print(f"\nStarting training ({epochs} epochs)...")
    model.fit(
        train_dataloader=train_samples,
        epochs=epochs,
        warmup_steps=100,
        output_path=output_dir,
        show_progress_bar=True,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print(f"\n✓ Training completed")
    print(f"  Model saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    train_reranker(epochs=epochs, batch_size=batch_size)
