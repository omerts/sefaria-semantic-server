"""
Build training pairs from Sefaria links
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import random


def load_normalized_entries(data_dir: str = "data/normalized") -> List[Dict[str, Any]]:
    """Loads all normalized entries"""
    entries = []
    normalized_dir = Path(data_dir)
    
    if not normalized_dir.exists():
        print(f"Directory not found: {normalized_dir}")
        return entries
    
    for json_file in normalized_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding='utf-8'))
            entries.extend(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return entries


def build_positive_pairs(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds positive pairs from Sefaria links
    
    Args:
        entries: List of entries
    
    Returns:
        List of pairs (idea, source, label=1)
    """
    pairs = []
    
    for entry in entries:
        idea_text = entry.get("text", "")
        if not idea_text or len(idea_text) < 10:
            continue
        
        links = entry.get("links", [])
        if not links:
            continue
        
        # Create positive pairs
        for link in links:
            target_ref = link.get("target_ref", "")
            if target_ref:
                pairs.append({
                    "idea": idea_text,
                    "source": target_ref,
                    "label": 1,
                    "entry_id": entry.get("id"),
                    "sefaria_ref": entry.get("sefaria_ref")
                })
    
    return pairs


def build_negative_pairs(entries: List[Dict[str, Any]], positive_pairs: List[Dict[str, Any]], 
                        num_negatives: int = None) -> List[Dict[str, Any]]:
    """
    Builds negative pairs (idea + unrelated source)
    
    Args:
        entries: List of entries
        positive_pairs: Positive pairs
        num_negatives: Number of negative pairs (if None, same as positive)
    
    Returns:
        List of negative pairs
    """
    if num_negatives is None:
        num_negatives = len(positive_pairs)
    
    # Collect all available refs
    all_refs = set()
    for entry in entries:
        ref = entry.get("sefaria_ref", "")
        if ref:
            all_refs.add(ref)
    
    # Create negative pairs
    negative_pairs = []
    ideas_used = set()
    
    for pair in positive_pairs:
        idea = pair["idea"]
        if idea in ideas_used:
            continue
        
        ideas_used.add(idea)
        
        # Choose random ref that's not related
        positive_refs = {p["source"] for p in positive_pairs if p["idea"] == idea}
        available_refs = list(all_refs - positive_refs)
        
        if available_refs:
            negative_ref = random.choice(available_refs)
            negative_pairs.append({
                "idea": idea,
                "source": negative_ref,
                "label": 0,
                "entry_id": pair.get("entry_id"),
                "sefaria_ref": pair.get("sefaria_ref")
            })
        
        if len(negative_pairs) >= num_negatives:
            break
    
    return negative_pairs


def build_training_pairs(output_file: str = "training/pairs.json",
                        data_dir: str = "data/normalized",
                        include_negatives: bool = True):
    """
    Builds training pairs from Sefaria links
    
    Args:
        output_file: Output file
        data_dir: Normalized data directory
        include_negatives: Whether to include negative pairs
    """
    print("Loading entries...")
    entries = load_normalized_entries(data_dir)
    print(f"Found {len(entries)} entries")
    
    print("Building positive pairs...")
    positive_pairs = build_positive_pairs(entries)
    print(f"Built {len(positive_pairs)} positive pairs")
    
    all_pairs = positive_pairs.copy()
    
    if include_negatives:
        print("Building negative pairs...")
        negative_pairs = build_negative_pairs(entries, positive_pairs)
        print(f"Built {len(negative_pairs)} negative pairs")
        all_pairs.extend(negative_pairs)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(
        json.dumps(all_pairs, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Positive: {len(positive_pairs)}")
    if include_negatives:
        print(f"  Negative: {len(negative_pairs)}")


if __name__ == "__main__":
    import sys
    
    include_negatives = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-negatives":
        include_negatives = False
    
    build_training_pairs(include_negatives=include_negatives)
