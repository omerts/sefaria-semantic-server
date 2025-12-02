"""
Text chunking by text type
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import sys
from pathlib import Path

# Handle both relative and absolute imports
try:
    from ingestion.schema import TextEntry, Chunk
except ImportError:
    # If running as script, use absolute import
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.schema import TextEntry, Chunk


# Hebrew number conversion
HEBREW_NUMBERS = [
    (400, "ת"),
    (300, "ש"),
    (200, "ר"),
    (100, "ק"),
    (90, "צ"),
    (80, "פ"),
    (70, "ע"),
    (60, "ס"),
    (50, "נ"),
    (40, "מ"),
    (30, "ל"),
    (20, "כ"),
    (10, "י"),
    (9, "ט"),
    (8, "ח"),
    (7, "ז"),
    (6, "ו"),
    (5, "ה"),
    (4, "ד"),
    (3, "ג"),
    (2, "ב"),
    (1, "א"),
]


def number_to_hebrew(n: int) -> str:
    """
    Converts a number to Hebrew numeral (1-999)

    Args:
        n: Number to convert (1-999)

    Returns:
        Hebrew numeral string
    """
    if n <= 0 or n > 999:
        raise ValueError("supported range: 1–999")

    result = ""
    num = n

    for value, letter in HEBREW_NUMBERS:
        while num >= value:
            result += letter
            num -= value

    # Avoid special cases like forming השם (e.g., 15, 16)
    if result == "יה":
        result = "טו"
    elif result == "יו":
        result = "טז"

    return result


def extract_chapter_verse(sefaria_ref: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extracts chapter and verse numbers from sefaria_ref
    Also handles Talmud format: BookName.2a:1 (daf+amud:segment)

    Args:
        sefaria_ref: Reference like "Genesis.1:1" or "Berakhot.2a:1"

    Returns:
        Tuple of (chapter/page, verse/segment) or (None, None) if not found
    """
    # Pattern for Talmud: BookName.2a:1 or BookName.2b:1
    talmud_match = re.search(r"\.(\d+)([ab]):(\d+)", sefaria_ref)
    if talmud_match:
        daf_number = int(talmud_match.group(1))
        amud = talmud_match.group(2)
        segment = int(talmud_match.group(3))
        # Convert daf+amud to page number: 2a=2, 2b=2, 3a=3, 3b=3, etc.
        # For display purposes, we'll use daf_number as page
        return daf_number, segment

    # Pattern: BookName.chapter:verse or BookName.chapter:verse:subverse
    match = re.search(r"\.(\d+):(\d+)", sefaria_ref)
    if match:
        chapter = int(match.group(1))
        verse = int(match.group(2))
        return chapter, verse
    return None, None


def add_book_specific_fields(
    chunk: Dict[str, Any], sefaria_ref: str, address_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Adds book-specific fields to chunk based on addressTypes metadata

    Uses addressTypes from Sefaria API to dynamically add only relevant fields.
    Only adds fields that exist in addressTypes, without null values.

    Args:
        chunk: Chunk dictionary
        sefaria_ref: Sefaria reference string (e.g., "Genesis.1:1", "Berakhot.2a:1")
        address_types: List of address types from Sefaria API (e.g., ["Perek", "Pasuk"])

    Returns:
        Updated chunk with only relevant addressType fields (no null values)
    """
    # If addressTypes is available, use it for dynamic mapping
    if address_types and len(address_types) > 0:
        # Parse sefaria_ref to extract values
        # Format: BookName.level1:level2:level3...
        # For Talmud: BookName.2a:1 (daf+amud:line)
        # For Tanakh: BookName.1:1 (chapter:verse)

        # Extract the part after the book name
        ref_parts = sefaria_ref.split(".", 1)
        if len(ref_parts) < 2:
            return chunk  # No sections in ref

        sections_str = ref_parts[1]

        # Parse sections based on addressTypes
        # Handle Talmud format: "2a:1" -> ["2a", "1"]
        # Handle regular format: "1:1" -> ["1", "1"]
        if ":" in sections_str:
            sections = sections_str.split(":")
        else:
            sections = [sections_str]

        # Map addressTypes to field names (lowercase, camelCase)
        field_mapping = {
            "perek": "perek",
            "pasuk": "verse",  # Pasuk maps to verse
            "daf": "daf",
            "talmud": "daf",  # Talmud also maps to daf
            "line": "line",
            "integer": "line",  # Integer often maps to line
            "mishnah": "mishnah",
            "siman": "siman",
            "seif": "seif",
            "chapter": "chapter",
        }

        # Add fields based on addressTypes
        for i, addr_type in enumerate(address_types):
            if i >= len(sections):
                break

            field_name = field_mapping.get(addr_type.lower(), addr_type.lower())
            section_value = sections[i]

            # Handle Talmud format: "2a" -> daf=2, amud="a"
            if addr_type.lower() in ["daf", "talmud"] and (
                "a" in section_value.lower() or "b" in section_value.lower()
            ):
                # Extract daf number and amud
                talmud_match = re.search(r"(\d+)([ab])", section_value.lower())
                if talmud_match:
                    chunk["daf"] = int(talmud_match.group(1))
                    chunk["amud"] = talmud_match.group(2)
                    # Also add masechet for Talmud
                    chunk["masechet"] = chunk.get("book", "")
                else:
                    # Try to parse as number
                    try:
                        chunk[field_name] = int(section_value)
                    except ValueError:
                        chunk[field_name] = section_value
            else:
                # Regular numeric or string value
                try:
                    # Try to convert to int if it's a number
                    chunk[field_name] = int(section_value)
                except ValueError:
                    chunk[field_name] = section_value

        # Handle special case: Shulchan Arukh part name
        book = chunk.get("book", "")
        if any(addr.lower() in ["siman", "seif"] for addr in address_types):
            # Extract part name from book name for Shulchan Arukh
            if "Orach Chayim" in book or "אורח חיים" in book:
                chunk["part"] = "Orach Chayim"
            elif "Yoreh Deah" in book or "יורה דעה" in book:
                chunk["part"] = "Yoreh Deah"
            elif "Even HaEzer" in book or "אבן העזר" in book:
                chunk["part"] = "Even HaEzer"
            elif "Choshen Mishpat" in book or "חושן משפט" in book:
                chunk["part"] = "Choshen Mishpat"

        return chunk

    # Fallback: old logic if addressTypes not available
    categories = chunk.get("category", [])
    book = chunk.get("book", "")

    # Check book type
    is_talmud = any(
        cat.lower() in ["talmud", "תלמוד", "bavli", "בבלי"]
        for cat in categories
        if isinstance(cat, str)
    )

    is_mishnah = any(
        cat.lower() in ["mishnah", "משנה"] for cat in categories if isinstance(cat, str)
    )

    is_tanakh = any(
        cat.lower() in ["tanakh", "תנ״ך", "torah", "תורה", "bible"]
        for cat in categories
        if isinstance(cat, str)
    )

    is_shulchan_arukh = (
        any(
            cat.lower() in ["halacha", "הלכה", "shulchan arukh", "שולחן ערוך", 'שו"ע']
            for cat in categories
            if isinstance(cat, str)
        )
        or "shulchan" in book.lower()
        or "שולחן" in book.lower()
    )

    # Only add fields for detected book type (no null values)
    if is_talmud:
        # For Talmud: masechet, daf, amud
        talmud_match = re.search(r"\.(\d+)([ab]):(\d+)", sefaria_ref)
        if talmud_match:
            daf_number = int(talmud_match.group(1))
            amud = talmud_match.group(2)
            chunk["masechet"] = book
            chunk["daf"] = daf_number
            chunk["amud"] = amud
    elif is_mishnah:
        # For Mishnah: perek, mishnah
        chapter, verse = extract_chapter_verse(sefaria_ref)
        if chapter is not None:
            chunk["perek"] = chapter
        if verse is not None:
            chunk["mishnah"] = verse
    elif is_shulchan_arukh:
        # For Shulchan Arukh: part, siman, seif
        chapter, verse = extract_chapter_verse(sefaria_ref)

        # Extract part name from book name
        part_name = "Orach Chayim"  # Default
        if "Orach Chayim" in book or "אורח חיים" in book:
            part_name = "Orach Chayim"
        elif "Yoreh Deah" in book or "יורה דעה" in book:
            part_name = "Yoreh Deah"
        elif "Even HaEzer" in book or "אבן העזר" in book:
            part_name = "Even HaEzer"
        elif "Choshen Mishpat" in book or "חושן משפט" in book:
            part_name = "Choshen Mishpat"

        chunk["part"] = part_name
        if chapter is not None:
            chunk["siman"] = chapter
        if verse is not None:
            chunk["seif"] = verse
    elif is_tanakh:
        # For Tanakh: chapter, verse
        chapter, verse = extract_chapter_verse(sefaria_ref)
        if chapter is not None:
            chunk["chapter"] = chapter
        if verse is not None:
            chunk["verse"] = verse

    return chunk


def chunk_tanakh(entry: TextEntry) -> List[Dict[str, Any]]:
    """Chunking for Tanakh - by verses or small groups"""
    text = entry.text
    chunks = []

    # Split by verses (verse mark: ׃ or number)
    verses = re.split(r"[׃\n]", text)

    # Filter out empty verses
    verses = [v.strip() for v in verses if v.strip()]

    # If only one verse after splitting, use the original ref
    # Otherwise, add sub-verse numbers
    if len(verses) == 1:
        # Single verse - use original ref without adding sub-number
        chunk = {
            "chunk_id": f"{entry.id}-verse0",
            "parent_id": entry.id,
            "sefaria_ref": entry.sefaria_ref,
            "book": entry.book,
            "category": entry.category,
            "text": verses[0],
            "position": 0,
            "chunk_type": "verse",
        }
        chunks.append(
            add_book_specific_fields(chunk, entry.sefaria_ref, entry.address_types)
        )
    else:
        # Multiple parts - add sub-verse numbers
        for i, verse in enumerate(verses):
            verse_ref = f"{entry.sefaria_ref}:{i+1}"
            chunk = {
                "chunk_id": f"{entry.id}-verse{i}",
                "parent_id": entry.id,
                "sefaria_ref": verse_ref,
                "book": entry.book,
                "category": entry.category,
                "text": verse,
                "position": i,
                "chunk_type": "verse",
            }
            chunks.append(
                add_book_specific_fields(chunk, verse_ref, entry.address_types)
            )

    return chunks if chunks else [create_default_chunk(entry, 0)]


def chunk_mishnah(entry: TextEntry) -> List[Dict[str, Any]]:
    """Chunking for Mishnah - complete mishnah unit as single chunk

    Mishnah should be kept as one complete unit to preserve full context.
    """
    chunk = {
        "chunk_id": f"{entry.id}-mishnah",
        "parent_id": entry.id,
        "sefaria_ref": entry.sefaria_ref,
        "book": entry.book,
        "category": entry.category,
        "text": entry.text,
        "position": 0,
        "chunk_type": "mishnah",
    }
    return [add_book_specific_fields(chunk, entry.sefaria_ref, entry.address_types)]


def _create_talmud_chunk(
    chunk_text: str,
    start_entry: TextEntry,
    chunks: List[Dict[str, Any]],
    chunk_type: str = "talmud",
) -> None:
    """Helper function to create and append a talmud chunk"""
    chunk = {
        "chunk_id": f"{start_entry.id}-{chunk_type}{len(chunks)}",
        "parent_id": start_entry.id,
        "sefaria_ref": start_entry.sefaria_ref,
        "book": start_entry.book,
        "category": start_entry.category,
        "text": chunk_text,
        "position": len(chunks),
        "chunk_type": chunk_type,
    }
    chunks.append(
        add_book_specific_fields(
            chunk, start_entry.sefaria_ref, start_entry.address_types
        )
    )


def chunk_talmud_page(entries: List[TextEntry]) -> List[Dict[str, Any]]:
    """
    Chunking for Talmud page - processes all entries together

    Rules:
    1. All lines until "גמ׳" = one mishnah chunk
    2. After "גמ׳", chunk by periods ".":
       - Split text by periods "."
       - If distance between periods < 30 chars, wait for next period
       - Continue accumulating until >= 30 chars or MAX_CHARS reached
    """
    if not entries:
        return []

    chunks = []
    MIN_CHARS = 30  # Minimum chars between periods
    MAX_CHARS = 200  # Maximum chars per chunk

    # Process all entries together - chunk by periods
    gemara_entries = entries

    # Combine all text
    gemara_text = " ".join(
        entry.text.strip() for entry in gemara_entries if entry.text.strip()
    )

    if gemara_text:
        # Split by periods, keeping the periods
        # Use regex to split but keep delimiters
        parts = re.split(r"(\.)", gemara_text)

        # Reconstruct sentences with periods
        sentences = []
        current_sentence = ""
        for part in parts:
            if part == ".":
                if current_sentence:
                    sentences.append(current_sentence + ".")
                    current_sentence = ""
            else:
                current_sentence += part

        # Add last sentence if it doesn't end with period
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # Now chunk sentences: each sentence is a chunk, but if chunk < MIN_CHARS, accumulate next sentence
        # If a single sentence > MAX_CHARS, split it by spaces to fit MAX_CHARS
        current_chunk_parts = []
        current_chunk_size = 0
        current_chunk_start_entry = gemara_entries[0] if gemara_entries else None

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # If sentence itself is > MAX_CHARS, split it into multiple chunks
            if sentence_len > MAX_CHARS:
                # Save current chunk first if exists
                if current_chunk_parts:
                    chunk_text = " ".join(current_chunk_parts)
                    if chunk_text:
                        _create_talmud_chunk(
                            chunk_text, current_chunk_start_entry, chunks
                        )

                # Split long sentence by spaces
                words = sentence.split()
                current_words = []
                current_words_len = 0

                for word in words:
                    word_len = len(word) + 1  # +1 for space
                    if current_words_len + word_len > MAX_CHARS and current_words:
                        # Save current chunk
                        chunk_text = " ".join(current_words)
                        if chunk_text:
                            _create_talmud_chunk(
                                chunk_text, current_chunk_start_entry, chunks
                            )
                        current_words = [word]
                        current_words_len = len(word)
                    else:
                        current_words.append(word)
                        current_words_len += word_len

                # Save last part of split sentence
                if current_words:
                    current_chunk_parts = [" ".join(current_words)]
                    current_chunk_size = len(current_chunk_parts[0])
                else:
                    current_chunk_parts = []
                    current_chunk_size = 0
                continue

            # Normal processing for sentences <= MAX_CHARS
            # If adding this sentence would exceed MAX_CHARS, save current chunk first
            if current_chunk_parts and current_chunk_size + sentence_len > MAX_CHARS:
                chunk_text = " ".join(current_chunk_parts)
                if chunk_text:
                    _create_talmud_chunk(chunk_text, current_chunk_start_entry, chunks)

                # Start new chunk with this sentence
                current_chunk_start_entry = (
                    gemara_entries[0] if gemara_entries else None
                )
                current_chunk_parts = [sentence]
                current_chunk_size = sentence_len
            elif not current_chunk_parts:
                # No current chunk - start new one with this sentence
                current_chunk_parts = [sentence]
                current_chunk_size = sentence_len
            elif current_chunk_size < MIN_CHARS:
                # Current chunk is < MIN_CHARS - add this sentence to reach MIN_CHARS
                current_chunk_parts.append(sentence)
                current_chunk_size += sentence_len
            else:
                # Current chunk is >= MIN_CHARS - save it and start new chunk with this sentence
                chunk_text = " ".join(current_chunk_parts)
                if chunk_text:
                    _create_talmud_chunk(chunk_text, current_chunk_start_entry, chunks)

                # Start new chunk with this sentence
                current_chunk_start_entry = (
                    gemara_entries[0] if gemara_entries else None
                )
                current_chunk_parts = [sentence]
                current_chunk_size = sentence_len

        # Save last chunk
        if current_chunk_parts:
            chunk_text = " ".join(current_chunk_parts)
            if chunk_text:
                _create_talmud_chunk(chunk_text, current_chunk_start_entry, chunks)

    if not chunks and entries:
        # Fallback: create default chunk from first entry
        return [create_default_chunk(entries[0], 0)]
    return chunks


def chunk_halacha(entry: TextEntry) -> List[Dict[str, Any]]:
    """Chunking for Halacha (Rambam, Shulchan Arukh) - by paragraphs/sections"""
    text = entry.text
    chunks = []

    # Split by sections (numbers or letters)
    # or by long lines
    parts = re.split(r"[\n]{2,}|(?=\d+\.)", text)

    for i, part in enumerate(parts):
        part = part.strip()
        if not part or len(part) < 10:
            continue

        chunk = {
            "chunk_id": f"{entry.id}-halacha{i}",
            "parent_id": entry.id,
            "sefaria_ref": entry.sefaria_ref,
            "book": entry.book,
            "category": entry.category,
            "text": part,
            "position": i,
            "chunk_type": "halacha",
        }
        chunks.append(
            add_book_specific_fields(chunk, entry.sefaria_ref, entry.address_types)
        )

    return chunks if chunks else [create_default_chunk(entry, 0)]


def chunk_text(entry: TextEntry) -> List[Dict[str, Any]]:
    """
    Chunks text by book type

    Args:
        entry: TextEntry to analyze

    Returns:
        List of chunks
    """
    book = entry.book.lower()
    categories = [c.lower() for c in entry.category]

    # Identify text type by category or book name
    if any(cat in ["tanakh", 'תנ"ך', "bible"] for cat in categories):
        return chunk_tanakh(entry)

    # Check for Mishnah
    is_mishnah = any(cat in ["mishnah", "משנה"] for cat in categories)

    if is_mishnah:
        return chunk_mishnah(entry)
    elif any(
        cat in ["halacha", "הלכה", "rambam", 'רמב"ם', "shulchan arukh", 'שו"ע']
        for cat in categories
    ):
        return chunk_halacha(entry)
    elif "mishneh torah" in book or "rambam" in book:
        return chunk_halacha(entry)
    elif "mishnah" in book:
        return chunk_mishnah(entry)
    else:
        # Default: assume gemara-like (can be split)
        return chunk_halacha(entry)


def create_default_chunk(entry: TextEntry, position: int) -> Dict[str, Any]:
    """Creates a default chunk"""
    chunk = {
        "chunk_id": f"{entry.id}-chunk{position}",
        "parent_id": entry.id,
        "sefaria_ref": entry.sefaria_ref,
        "book": entry.book,
        "category": entry.category,
        "text": entry.text,
        "position": position,
        "chunk_type": "default",
    }
    return add_book_specific_fields(chunk, entry.sefaria_ref, entry.address_types)


def process_all_entries(
    data_dir: str = "data/normalized",
    output_dir: str = "data/chunks",
    books: List[str] = None,
):
    """Processes all entries and creates chunks"""
    import json
    from pathlib import Path
    from typing import List

    # Handle imports
    try:
        from ingestion.schema import TextEntry
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ingestion.schema import TextEntry

    normalized_dir = Path(data_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not normalized_dir.exists():
        print(f"Directory not found: {normalized_dir}")
        return

    output_path.mkdir(exist_ok=True, parents=True)

    json_files = list(normalized_dir.glob("*.json"))

    # Filter by books if provided
    if books:
        # Normalize book names for comparison (handle underscores, etc.)
        books_normalized = {book.replace("/", "_") for book in books}
        json_files = [f for f in json_files if f.stem in books_normalized]
        if not json_files:
            print(f"No matching files found for books: {books}")
            return

    print(f"\nChunking files from: {normalized_dir}")
    print(f"Output directory: {output_path}")
    print(f"Processing {len(json_files)} files...\n")

    all_chunks = []

    for i, json_file in enumerate(json_files, 1):
        book = json_file.stem
        print(f"[{i}/{len(json_files)}] Processing: {book}")
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            entries = [TextEntry(**item) for item in data]

            book_chunks = []

            # Check if this is a Talmud book - need special processing
            is_talmud = any(
                entry.category
                and any(
                    cat.lower() in ["talmud", "תלמוד", "bavli", "בבלי"]
                    for cat in entry.category
                )
                for entry in entries[:5]  # Check first few entries
            )

            if is_talmud:
                # Group entries by page (daf) for Talmud
                # Entries are already sorted, so process them in order
                current_page_entries = []
                current_daf = None

                for entry in entries:
                    # Extract daf from sefaria_ref: "BookName.2a:1" -> "2a"
                    ref_parts = entry.sefaria_ref.split(".")
                    if len(ref_parts) >= 2:
                        daf_part = ref_parts[1].split(":")[0]  # "2a:1" -> "2a"

                        # If daf changed, process previous page
                        if current_daf is not None and daf_part != current_daf:
                            page_chunks = chunk_talmud_page(current_page_entries)
                            book_chunks.extend(page_chunks)
                            current_page_entries = []

                        current_daf = daf_part
                        current_page_entries.append(entry)

                # Process last page
                if current_page_entries:
                    page_chunks = chunk_talmud_page(current_page_entries)
                    book_chunks.extend(page_chunks)
            else:
                # Regular processing - one entry at a time
                for entry in entries:
                    chunks = chunk_text(entry)
                    book_chunks.extend(chunks)

            # Save for each book separately
            book_output_path = output_path / f"{book}.json"
            book_output_path.write_text(
                json.dumps(book_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            abs_path = book_output_path.resolve()
            file_size = book_output_path.stat().st_size
            all_chunks.extend(book_chunks)
            print(f"  ✓ Saved: {abs_path}")
            print(f"    Chunks: {len(book_chunks)}, File size: {file_size} bytes")

        except Exception as e:
            print(f"  ✗ Error processing {book}: {e}")

    # Save all chunks in one file
    all_chunks_path = output_path / "all_chunks.json"
    all_chunks_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    abs_path = all_chunks_path.resolve()
    file_size = all_chunks_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"Chunking Summary:")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  All chunks file: {abs_path}")
    print(f"  File size: {file_size} bytes")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_all_entries()
