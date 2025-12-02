from pydantic import BaseModel
from typing import List, Optional


class TextEntry(BaseModel):
    """Data structure for a Sefaria text entry"""

    id: str
    sefaria_ref: str
    book: str
    category: List[str]
    text: str
    links: List[dict]
    address_types: Optional[List[str]] = (
        None  # addressTypes from Sefaria API (e.g., ["Perek", "Pasuk"])
    )


class Chunk(BaseModel):
    """Data structure for a text chunk"""

    chunk_id: str
    parent_id: str
    sefaria_ref: str
    book: str
    category: List[str]
    text: str
    position: int
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None
