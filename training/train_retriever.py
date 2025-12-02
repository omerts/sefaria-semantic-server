"""
Train Retriever Model
"""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from typing import List


def load_pairs(pairs_file: str = "training/pairs.json") -> List[InputExample]:
    """
    Loads training pairs
    
    Args:
        pairs_file: Pairs file
    
    Returns:
        List of InputExample
    """
    pairs_path = Path(pairs_file)
    if not pairs_path.exists():
        raise FileNotFoundError(f"File not found: {pairs_path}")
    
    data = json.loads(pairs_path.read_text(encoding='utf-8'))
    
    examples = []
    for pair in data:
        # Only positive pairs for retriever training (MultipleNegativesRankingLoss)
        if pair.get("label", 0) == 1:
            examples.append(
                InputExample(texts=[pair["idea"], pair["source"]])
            )
    
    return examples


def train_retriever(
    base_model: str = "BAAI/bge-base-en-v1.5",
    pairs_file: str = "training/pairs.json",
    output_dir: str = "models/retriever",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5
):
    """
    Trains retriever model
    
    Args:
        base_model: Base model
        pairs_file: Training pairs file
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    print(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    
    print("Loading training pairs...")
    train_examples = load_pairs(pairs_file)
    print(f"Found {len(train_examples)} training examples")
    
    if not train_examples:
        print("⚠ No training examples!")
        return
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Define loss function
    # MultipleNegativesRankingLoss is suitable for retrieval
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Training
    print(f"\nStarting training ({epochs} epochs)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=output_dir,
        show_progress_bar=True
    )
    
    print(f"\n✓ Training completed")
    print(f"  Model saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    
    train_retriever(epochs=epochs, batch_size=batch_size)
