"""
Re-Ranker Model - ranks results and improves accuracy
"""

# Import SSL configuration FIRST - this configures SSL once for all modules
from utils.ssl_config import configure_ssl

configure_ssl()

from typing import List, Tuple, Optional
from pathlib import Path
import os
import re
import unicodedata


def normalize_hebrew(text: str) -> str:
    """
    Basic normalization for Hebrew:
    - Lowercase
    - Remove niqqud (vowel marks)
    - Remove punctuation
    - Collapse multiple spaces
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # Remove niqqud (combining marks)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # Lowercase
    text = text.lower()

    # Replace punctuation with space (keep Hebrew letters)
    text = re.sub(r"[^\w\u05d0-\u05ea]+", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def lexical_overlap_score(query: str, text: str) -> float:
    """
    Simple lexical overlap: how many query tokens appear in the text.
    Score ~ recall: |intersection| / |query_tokens|
    """
    q_norm = normalize_hebrew(query)
    t_norm = normalize_hebrew(text)

    if not q_norm or not t_norm:
        return 0.0

    q_tokens = set(q_norm.split())
    t_tokens = set(t_norm.split())

    if not q_tokens or not t_tokens:
        return 0.0

    inter = q_tokens.intersection(t_tokens)
    return len(inter) / len(q_tokens)


def exact_match_bonus(query: str, text: str) -> float:
    """
    Bonus if normalized query string appears as substring in text.
    For short queries this is very strong.
    """
    q_norm = normalize_hebrew(query)
    t_norm = normalize_hebrew(text)

    if not q_norm or not t_norm:
        return 0.0

    if q_norm and q_norm in t_norm:
        return 1.0
    return 0.0


class ReRanker:
    """Class for re-ranker - re-ranking results"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to trained model. If None, uses base model
        """
        self.model_path = model_path
        self.model = None
        self.base_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._load_model()

    def _load_model(self):
        """Loads the model"""
        try:
            # SSL is already configured by utils.ssl_config when module is imported
            from sentence_transformers import CrossEncoder

            if self.model_path and Path(self.model_path).exists():
                print(f"Loading trained model: {self.model_path}")
                self.model = CrossEncoder(self.model_path)
            else:
                print(f"Loading base model: {self.base_model_name}")
                self.model = CrossEncoder(self.base_model_name)

            print("✓ Re-Ranker loaded")
        except ImportError:
            print(
                "⚠ sentence-transformers not installed. Install: pip install sentence-transformers"
            )
            self.model = None
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("💡 Tip: Set DISABLE_SSL_VERIFY=1 to disable SSL verification")
            self.model = None

    def _semantic_scores(self, query: str, texts: List[str]) -> List[float]:
        """
        Raw semantic scores from cross-encoder.
        """
        if not self.model or not texts:
            return [0.0] * len(texts)

        try:
            # Create pairs (query, text)
            pairs = [(query, text) for text in texts]

            # Calculate scores
            scores = self.model.predict(pairs)

            # Convert to python list
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = list(scores)

            return scores
        except Exception as e:
            print(f"Error getting semantic scores: {e}")
            return [0.0] * len(texts)

    def score(self, query: str, texts: List[str]) -> List[float]:
        """
        Hybrid score combining semantic, lexical, and exact match:
        final = w_sem * semantic + w_lex * lexical + w_exact * exact_match

        Args:
            query: Query
            texts: List of texts to rank

        Returns:
            List of scores (higher = more relevant)
        """
        if not texts:
            return []

        # 1. Semantic scores (cross-encoder)
        sem_scores = self._semantic_scores(query, texts)

        # Normalize semantic scores to 0..1 per batch (min-max)
        if sem_scores:
            s_min = min(sem_scores)
            s_max = max(sem_scores)
            denom = (s_max - s_min) or 1.0
            sem_norm = [(s - s_min) / denom for s in sem_scores]
        else:
            sem_norm = [0.0] * len(texts)

        # 2. Lexical + exact match scores
        lex_scores: List[float] = []
        exact_scores: List[float] = []

        for t in texts:
            lex_scores.append(lexical_overlap_score(query, t))
            exact_scores.append(exact_match_bonus(query, t))

        # 3. Weights - can be tuned
        w_sem = 0.7  # semantic (cross-encoder)
        w_lex = 0.2  # lexical overlap
        w_exact = 0.4  # strong bonus for exact match

        # 4. Combine scores
        final_scores: List[float] = []
        for s, l, e in zip(sem_norm, lex_scores, exact_scores):
            # Clamp values to [0, 1] just in case
            s = max(0.0, min(1.0, s))
            l = max(0.0, min(1.0, l))
            e = max(0.0, min(1.0, e))

            score = w_sem * s + w_lex * l + w_exact * e
            final_scores.append(score)

        return final_scores

    def rank(
        self, query: str, texts: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Ranks and returns sorted results

        Args:
            query: Query
            texts: List of texts
            top_k: Number of top results (if None, returns all)

        Returns:
            List of tuples (text, score) sorted by score
        """
        scores = self.score(query, texts)

        # Sort by score
        ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)

        if top_k:
            return ranked[:top_k]

        return ranked
