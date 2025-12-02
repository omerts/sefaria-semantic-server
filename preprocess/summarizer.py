"""
Generate summaries for chunks (optional)
"""
from typing import Optional
import os


class Summarizer:
    """Class for generating summaries"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Args:
            model_name: Model name for summarization
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Loads the model (lazy loading)"""
        try:
            from transformers import pipeline
            print(f"Loading summarization model: {self.model_name}...")
            self.model = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=-1  # CPU
            )
            print("✓ Model loaded")
        except Exception as e:
            print(f"⚠ Cannot load summarization model: {e}")
            print("Summaries will not be generated")
            self.model = None
    
    def summarize(self, text: str, max_length: int = 40, min_length: int = 5) -> Optional[str]:
        """
        Creates a short summary of text
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
        
        Returns:
            Summary or None if failed
        """
        if not self.model or not text:
            return None
        
        try:
            # For Hebrew, English models don't always work well
            # Can use Hebrew-specific models
            if len(text) < min_length:
                return text[:max_length]
            
            result = self.model(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=2
            )
            
            if result and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            
            return None
        except Exception as e:
            print(f"Error creating summary: {e}")
            return None
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> list:
        """
        Extracts keywords (simple - by frequency)
        
        Args:
            text: Text to extract from
            num_keywords: Number of keywords
        
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Remove basic Hebrew stop words
        stop_words = {'את', 'של', 'על', 'אל', 'כי', 'אם', 'כל', 'זה', 'הוא', 'היא', 'הם', 'הן'}
        
        words = text.split()
        word_freq = {}
        
        for word in words:
            word = word.strip('.,;:()[]{}"\'')
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:num_keywords]]


# Global instance
_summarizer_instance = None


def get_summarizer() -> Summarizer:
    """Returns summarizer instance (singleton)"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = Summarizer()
    return _summarizer_instance


def summarize_chunk(text: str) -> Optional[str]:
    """Convenience function for creating summary"""
    summarizer = get_summarizer()
    return summarizer.summarize(text)


if __name__ == "__main__":
    # Test
    summarizer = Summarizer()
    test_text = "ברכה ראשונה היא ברכת המזון, ברכה שנייה היא ברכת התורה, ברכה שלישית היא ברכת כהנים"
    summary = summarizer.summarize(test_text)
    keywords = summarizer.extract_keywords(test_text)
    
    print(f"Text: {test_text}")
    print(f"Summary: {summary}")
    print(f"Keywords: {keywords}")
