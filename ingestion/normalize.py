"""
Normalization and processing of raw data from Sefaria

addressTypes - מיפוי סוגי כתובות ליצירת ref:

addressTypes מגדיר איך לפרמט כל רמה ב-Sefaria reference.
זהו מערך של מחרוזות, כל אחת מתאימה לרמה אחת במבנה ההיררכי.

סוגי addressTypes נפוצים:
- "Talmud" / "Daf" → דף עם אות עברית (2a, 2b, 3a, 3b...)
  * דורש context עם 'amud': 'a' או 'b'
  * דוגמה: ["Talmud", "Integer"] → Berakhot.2a:1

- "Integer" → מספר רגיל (1, 2, 3...)
  * משמש לרמות פשוטות כמו שורות, פסוקים
  * דוגמה: ["Perek", "Integer"] → Genesis.1:1

- "Perek" → פרק (1, 2, 3...)
  * משמש לתנ"ך, משנה
  * דוגמה: ["Perek", "Pasuk"] → Genesis.1:1

- "Pasuk" → פסוק (1, 2, 3...)
  * משמש לתנ"ך
  * דוגמה: ["Perek", "Pasuk"] → Genesis.1:1

- "Line" → שורה (1, 2, 3...)
  * משמש לתלמוד, שורות בדף
  * דוגמה: ["Daf", "Line"] → Berakhot.2a:1

- "Siman" → סימן (1, 2, 3...)
  * משמש לשולחן ערוך
  * דוגמה: ["Siman", "Seif"] → Shulchan Arukh, Orach Chayim.34:1

- "Seif" → סעיף (1, 2, 3...)
  * משמש לשולחן ערוך
  * דוגמה: ["Siman", "Seif"] → Shulchan Arukh, Orach Chayim.34:1

אם אין מיפוי ידוע ל-addressType, הקוד ישתמש באות עברית (א, ב, ג...)
עבור ערכים קטנים (1-22), או במספר רגיל אחרת.

שימוש ב-context:
- עבור "Talmud" / "Daf": צריך context עם 'amud': 'a' או 'b'
  * דוגמה: _build_ref([2, 1], context=[{'amud': 'a'}, {}]) → "Berakhot.2a:1"
- עבור סוגים אחרים: context ריק או לא נדרש

דוגמאות:
- Tanakh: addressTypes=["Perek", "Pasuk"] → Genesis.1:1
- Talmud: addressTypes=["Talmud", "Integer"] → Berakhot.2a:1 (עם context={'amud': 'a'})
- Mishnah: addressTypes=["Perek", "Mishnah"] → Mishnah Berakhot.1:1
- Shulchan Arukh: addressTypes=["Siman", "Seif"] → Shulchan Arukh, Orach Chayim.34:1

קישור לתיעוד Sefaria API:
https://www.sefaria.org/api/v3/texts/{tref}
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import sys

# Handle both relative and absolute imports
try:
    from .schema import TextEntry
except ImportError:
    # If running as script, use absolute import
    sys.path.insert(0, str(Path(__file__).parent))
    from schema import TextEntry


def load_raw(book: str, data_dir: str = "data/raw") -> Dict[str, Any]:
    """Loads a raw JSON file"""
    file_path = Path(data_dir) / f"{book.replace('/', '_')}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return json.loads(file_path.read_text(encoding="utf-8"))


def remove_nikud(text: str) -> str:
    """Removes nikud (vowel marks) from Hebrew text"""
    # Hebrew nikud: אָ אַ אֵ אֶ אִ אֹ אֻ אוּ אִי אֵי etc.
    nikud_pattern = re.compile(r"[\u0591-\u05C7]")
    return nikud_pattern.sub("", text)


def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if not text:
        return ""

    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", "", text)

    # Remove HTML entities: &nbsp; (non-breaking space), &thinsp; (thin space)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&thinsp;", " ")
    # Also handle numeric entities
    text = text.replace("&#160;", " ")  # &nbsp; numeric
    text = text.replace("&#8201;", " ")  # &thinsp; numeric

    # Replace hyphen/dash (מקף) with space BEFORE removing nikud
    # Hebrew maqaf (\u05BE) and regular hyphen (-)
    # Handle both uppercase and lowercase Unicode
    text = text.replace("\u05be", " ")  # Hebrew maqaf (uppercase)
    text = text.replace("\u05be", " ")  # Hebrew maqaf (lowercase)
    text = text.replace("-", " ")  # Regular hyphen
    # Also handle other dash types
    text = text.replace("–", " ")  # En dash
    text = text.replace("—", " ")  # Em dash

    # Remove nikud
    text = remove_nikud(text)

    # Clean double spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_links(raw_data: Dict[str, Any]) -> List[dict]:
    """Extracts links from raw data"""
    links = []

    # Sefaria stores links in different places
    if "links" in raw_data:
        links.extend(raw_data["links"])

    if "refs" in raw_data:
        for ref in raw_data["refs"]:
            links.append({"target_ref": ref})

    return links


class BaseNormalizer:
    """Base class for normalizers"""

    def __init__(self, raw: Dict[str, Any], book: str):
        self.raw = raw
        self.book = book
        self.links = extract_links(raw) if isinstance(raw, dict) else []
        self.categories = raw.get("categories", []) if isinstance(raw, dict) else []
        if not self.categories and isinstance(raw, dict) and "index" in raw:
            self.categories = raw["index"].get("categories", [])
        self.base_ref = book
        if "index" in raw and isinstance(raw, dict):
            self.base_ref = raw["index"].get("title", book)

        # Extract mapping metadata
        self.section_names = (
            raw.get("sectionNames", []) if isinstance(raw, dict) else []
        )
        self.address_types = (
            raw.get("addressTypes", []) if isinstance(raw, dict) else []
        )
        self.text_depth = raw.get("textDepth", 2) if isinstance(raw, dict) else 2
        self.lengths = raw.get("lengths", []) if isinstance(raw, dict) else []

    def normalize(self) -> List[TextEntry]:
        """Normalize the book - to be implemented by subclasses"""
        raise NotImplementedError

    def _clean_text(self, text: Any) -> Optional[str]:
        """Helper to clean and extract text"""
        if isinstance(text, list):
            text = text[0] if len(text) > 0 else None
        elif not isinstance(text, str) and text is not None:
            text = str(text)
        return clean_text(text) if text else None

    def _get_section_name(self, level: int, use_hebrew: bool = False) -> str:
        """Get section name for a given level (0-indexed)

        Args:
            level: Level index (0-based)
            use_hebrew: If True, prefer Hebrew names from addressTypes

        Returns:
            Section name (e.g., "Perek", "Pasuk", "Chapter", "Verse")
        """
        if use_hebrew and self.address_types and level < len(self.address_types):
            return self.address_types[level]
        elif self.section_names and level < len(self.section_names):
            return self.section_names[level]
        return f"Level{level + 1}"

    def _format_address_component(
        self, address_type: str, value: int, context: Dict[str, Any] = None
    ) -> str:
        """
        Format a single address component based on addressTypes

        addressTypes מגדיר איך לפרמט כל רמה ב-ref:
        - "Talmud" → דף עם אות (2a, 2b) - משתמש ב-context['amud'] אם קיים
        - "Integer" → מספר רגיל (5)
        - "Perek" → פרק (1)
        - "Pasuk" → פסוק (1)
        - "Daf" → דף (2) - אם יש amud, יהפוך ל-2a/2b
        - "Line" → שורה (1)
        - "Siman" → סימן (34)
        - "Seif" → סעיף (1)

        אם אין מיפוי ידוע, משתמש באות עברית (א, ב, ג...) או מספר

        Args:
            address_type: Type from addressTypes array (e.g., "Talmud", "Integer", "Perek")
            value: Numeric value (1-indexed)
            context: Optional context dict with additional info (e.g., {'amud': 'a'})

        Returns:
            Formatted string (e.g., "2a", "5", "1")
        """
        context = context or {}

        # Talmud/Daf format: add amud (a/b) if available
        if address_type.lower() in ["talmud", "daf"]:
            amud = context.get("amud", "")
            if amud:
                return f"{value}{amud}"
            # If no amud in context, check if we can infer from value
            # For Talmud, even indices might be 'a', odd might be 'b'
            # But better to have explicit context
            return str(value)

        # Integer, Perek, Pasuk, Line, Siman, Seif - simple number
        if address_type.lower() in [
            "integer",
            "perek",
            "pasuk",
            "line",
            "siman",
            "seif",
            "mishnah",
        ]:
            return str(value)

        # Unknown type - use Hebrew letter (א, ב, ג...) or number
        # If value is small (1-22), use Hebrew letter as fallback
        # זה עוזר במקרים שבהם אין מיפוי ידוע
        if 1 <= value <= 22:
            hebrew_letters = "אבגדהוזחטיכלמנסעפצקרשת"
            return hebrew_letters[value - 1]

        # For larger values, use number
        return str(value)

    def _build_ref(
        self, indices: List[int], context: List[Dict[str, Any]] = None
    ) -> str:
        """
        Build Sefaria reference from indices using addressTypes metadata

        משתמש ב-addressTypes כדי לדעת איך לפרמט כל רמה:
        - ["Perek", "Pasuk"] → Genesis.1:1
        - ["Talmud", "Integer"] → Berakhot.2a:1 (אם context[0]['amud'] = 'a')
        - ["Daf", "Line"] → Berakhot.2a:1

        Args:
            indices: List of numeric indices (1-indexed) for each level
            context: Optional list of context dicts, one per level
                    (e.g., [{'amud': 'a'}, {}] for Talmud)

        Returns:
            Sefaria reference string (e.g., "Genesis.1:1", "Berakhot.2a:1")
        """
        if not self.address_types or len(indices) != len(self.address_types):
            # Fallback: simple format with colons
            ref_parts = [str(idx) for idx in indices]
            return f"{self.base_ref}.{':'.join(ref_parts)}"

        context = context or [{}] * len(indices)

        # Format each component based on its addressType
        ref_parts = []
        for i, (idx, addr_type) in enumerate(zip(indices, self.address_types)):
            level_context = context[i] if i < len(context) else {}
            formatted = self._format_address_component(addr_type, idx, level_context)
            ref_parts.append(formatted)

        return f"{self.base_ref}.{':'.join(ref_parts)}"


class TalmudNormalizer(BaseNormalizer):
    """Normalizer for Talmud (Bavli)

    Uses addressTypes metadata: ["Daf", "Line"] for dynamic mapping
    Structure: text[i] = page (daf+amud), text[i][j] = segment (line)
    """

    def normalize(self) -> List[TextEntry]:
        """Normalize Talmud structure using addressTypes metadata"""
        entries = []

        # Verify we have the expected structure (2 levels: Daf/Line)
        if self.text_depth != 2:
            print(f"⚠ Warning: Expected textDepth=2 for Talmud, got {self.text_depth}")

        # Get English pages (text structure)
        en_chapters = []
        if "text" in self.raw and isinstance(self.raw["text"], list):
            en_chapters = self.raw["text"]

        # Get Hebrew pages
        he_chapters = []
        if "he" in self.raw and isinstance(self.raw["he"], list):
            he_chapters = self.raw["he"]

        all_pages = en_chapters if en_chapters else []
        entry_idx = 0

        # Use lengths metadata if available
        expected_pages = (
            self.lengths[0] if self.lengths and len(self.lengths) > 0 else None
        )

        for i, page_segments in enumerate(all_pages):
            # Skip first two empty pages and empty pages
            if i < 2 or not page_segments:
                continue

            # Calculate daf (page) and amud (side)
            daf_index = i - 2
            daf_number = 2 + (daf_index // 2)
            amud = "a" if (daf_index % 2 == 0) else "b"

            # Get Hebrew segments for this page
            he_segments = []
            if i < len(he_chapters):
                he_segments = he_chapters[i]
                if not isinstance(he_segments, list):
                    he_segments = [he_segments] if he_segments else []

            # Iterate through segments in this page
            max_segments = max(len(page_segments), len(he_segments))

            for j in range(max_segments):
                segment_index = j + 1

                # Get English segment
                seg_text = None
                if j < len(page_segments):
                    seg_text = page_segments[j]

                # Get Hebrew segment
                he_seg = None
                if j < len(he_segments):
                    he_seg = he_segments[j]

                # Clean text
                he_seg = self._clean_text(he_seg)
                seg_text = self._clean_text(seg_text)

                # Skip if no text at all
                if (not he_seg or not he_seg.strip()) and (
                    not seg_text or not seg_text.strip()
                ):
                    continue

                # Build ref using addressTypes metadata with context
                # Talmud addressTypes: ["Talmud", "Integer"] or ["Daf", "Line"]
                # Context provides amud ('a' or 'b') for the first level
                sefaria_ref = self._build_ref(
                    [daf_number, segment_index], context=[{"amud": amud}, {}]
                )

                entry = TextEntry(
                    id=f"{self.book.replace('/', '_')}.{entry_idx}",
                    sefaria_ref=sefaria_ref,
                    book=self.book,
                    category=self.categories,
                    text=he_seg if he_seg else "",
                    links=self.links if entry_idx == 0 else [],
                    address_types=self.address_types,
                )
                entries.append(entry)
                entry_idx += 1

        return entries


class TanakhNormalizer(BaseNormalizer):
    """Normalizer for Tanakh (Bible)

    Structure: Uses addressTypes metadata for dynamic mapping
    For Tanakh: addressTypes = ["Perek", "Pasuk"] -> text[i] = chapter, text[i][j] = verse
    """

    def normalize(self) -> List[TextEntry]:
        """Normalize Tanakh structure using addressTypes metadata"""
        entries = []

        # Verify we have the expected structure (2 levels: Perek/Pasuk)
        if self.text_depth != 2:
            print(f"⚠ Warning: Expected textDepth=2 for Tanakh, got {self.text_depth}")

        # Get Hebrew chapters
        he_chapters = []
        if "he" in self.raw and isinstance(self.raw["he"], list):
            he_chapters = self.raw["he"]

        # Get English chapters (for reference, not used in text field)
        en_chapters = []
        if "text" in self.raw and isinstance(self.raw["text"], list):
            en_chapters = self.raw["text"]
        elif (
            "text" in self.raw
            and isinstance(self.raw["text"], dict)
            and "en" in self.raw["text"]
        ):
            en_raw = self.raw["text"]["en"]
            if isinstance(en_raw, list):
                en_chapters = en_raw

        # If no Hebrew chapters, try text structure
        if not he_chapters and isinstance(self.raw.get("text"), list):
            text_segments = self.raw["text"]
            for item in text_segments:
                if isinstance(item, list):
                    he_chapters.append([s for s in item if isinstance(s, str)])

        entry_idx = 0

        # Use lengths metadata if available to validate structure
        expected_chapters = (
            self.lengths[0] if self.lengths and len(self.lengths) > 0 else None
        )

        # Iterate through chapters (i = chapter_index, 0-based)
        for chapter_idx, he_verses in enumerate(he_chapters):
            # Skip empty chapters
            if not he_verses or (isinstance(he_verses, list) and len(he_verses) == 0):
                continue

            # Chapter number = chapter_index + 1 (1-indexed)
            chapter_num = chapter_idx + 1

            # Get corresponding English verses for this chapter
            en_verses = []
            if chapter_idx < len(en_chapters):
                en_verses = en_chapters[chapter_idx]
                if not isinstance(en_verses, list):
                    en_verses = [en_verses] if en_verses else []

            # Skip if English chapter is also empty
            if not en_verses and not he_verses:
                continue

            # Iterate through verses in this chapter (j = verse_index, 0-based)
            for verse_idx, he_verse in enumerate(he_verses):
                # Verse number = verse_index + 1 (1-indexed)
                verse_num = verse_idx + 1

                # Get corresponding English verse
                en_verse = None
                if verse_idx < len(en_verses):
                    en_verse = en_verses[verse_idx]

                # Clean text
                he_verse = self._clean_text(he_verse)
                en_verse = self._clean_text(en_verse)

                # Skip if no Hebrew text
                if not he_verse or not he_verse.strip():
                    continue

                # Build ref using metadata-aware method
                # For Tanakh: Genesis.1:1 (Perek:Pasuk format)
                sefaria_ref = self._build_ref([chapter_num, verse_num])

                entry = TextEntry(
                    id=f"{self.book.replace('/', '_')}.{entry_idx}",
                    sefaria_ref=sefaria_ref,
                    book=self.book,
                    category=self.categories,
                    text=he_verse,
                    links=self.links if entry_idx == 0 else [],
                    address_types=self.address_types,
                )
                entries.append(entry)
                entry_idx += 1

        return entries


class MishnahNormalizer(BaseNormalizer):
    """Normalizer for Mishnah

    Uses addressTypes metadata: typically ["Chapter", "Mishnah"] or ["Perek", "Mishnah"]
    Structure: text[i] = chapter (perek), text[i][j] = mishnah
    """

    def normalize(self) -> List[TextEntry]:
        """Normalize Mishnah structure using addressTypes metadata"""
        entries = []

        # Verify we have the expected structure (2 levels: Perek/Mishnah)
        if self.text_depth != 2:
            print(f"⚠ Warning: Expected textDepth=2 for Mishnah, got {self.text_depth}")

        # Get chapters (perakim)
        chapters = []
        if "text" in self.raw and isinstance(self.raw["text"], list):
            chapters = self.raw["text"]

        # Get Hebrew chapters
        he_chapters = []
        if "he" in self.raw and isinstance(self.raw["he"], list):
            he_chapters = self.raw["he"]

        entry_idx = 0

        # Use lengths metadata if available
        expected_perakim = (
            self.lengths[0] if self.lengths and len(self.lengths) > 0 else None
        )

        # Iterate through chapters (perakim)
        for perek_idx, mishnayot in enumerate(chapters):
            # Skip empty chapters
            if not mishnayot or (isinstance(mishnayot, list) and len(mishnayot) == 0):
                continue

            perek_num = perek_idx + 1  # Chapter number (1-indexed)

            # Get Hebrew mishnayot for this chapter
            he_mishnayot = []
            if perek_idx < len(he_chapters):
                he_mishnayot = he_chapters[perek_idx]
                if not isinstance(he_mishnayot, list):
                    he_mishnayot = [he_mishnayot] if he_mishnayot else []

            # Iterate through mishnayot in this chapter
            max_mishnayot = max(len(mishnayot), len(he_mishnayot))

            for mishnah_idx in range(max_mishnayot):
                mishnah_num = mishnah_idx + 1

                # Get English mishnah
                mishnah_text = None
                if mishnah_idx < len(mishnayot):
                    mishnah_text = mishnayot[mishnah_idx]

                # Get Hebrew mishnah
                he_mishnah = None
                if mishnah_idx < len(he_mishnayot):
                    he_mishnah = he_mishnayot[mishnah_idx]

                # Clean text
                he_mishnah = self._clean_text(he_mishnah)
                mishnah_text = self._clean_text(mishnah_text)

                # Skip if no text at all
                if (not he_mishnah or not he_mishnah.strip()) and (
                    not mishnah_text or not mishnah_text.strip()
                ):
                    continue

                # Build ref using metadata-aware method
                # For Mishnah: Mishnah Berakhot.1:1 (Perek:Mishnah format)
                sefaria_ref = self._build_ref([perek_num, mishnah_num])

                entry = TextEntry(
                    id=f"{self.book.replace('/', '_')}.{entry_idx}",
                    sefaria_ref=sefaria_ref,
                    book=self.book,
                    category=self.categories,
                    text=he_mishnah if he_mishnah else "",
                    links=self.links if entry_idx == 0 else [],
                    address_types=self.address_types,
                )
                entries.append(entry)
                entry_idx += 1

        return entries


class ShulchanArukhNormalizer(BaseNormalizer):
    """Normalizer for Shulchan Arukh

    Uses addressTypes metadata: typically ["Siman", "Seif"]
    Structure: text[i] = siman (סימן), text[i][j] = seif (סעיף)
    For multi-part books: text[i] = part, text[i][j] = siman, text[i][j][k] = seif
    """

    def normalize(self) -> List[TextEntry]:
        """Normalize Shulchan Arukh structure using addressTypes metadata"""
        entries = []

        # Verify structure depth (usually 2 levels: Siman/Seif, but can be 3 for multi-part)
        if self.text_depth not in [2, 3]:
            print(
                f"⚠ Warning: Expected textDepth=2 or 3 for Shulchan Arukh, got {self.text_depth}"
            )

        # Get text structure
        text_data = []
        if "text" in self.raw and isinstance(self.raw["text"], list):
            text_data = self.raw["text"]

        # Get Hebrew text
        he_data = []
        if "he" in self.raw and isinstance(self.raw["he"], list):
            he_data = self.raw["he"]

        # Determine part name from book title or index
        part_name = "Orach Chayim"  # Default
        if "index" in self.raw and isinstance(self.raw, dict):
            index_data = self.raw["index"]
            # Try to get part name from title
            title = index_data.get("title", "")
            if "Orach Chayim" in title:
                part_name = "Orach Chayim"
            elif "Yoreh Deah" in title:
                part_name = "Yoreh Deah"
            elif "Even HaEzer" in title:
                part_name = "Even HaEzer"
            elif "Choshen Mishpat" in title:
                part_name = "Choshen Mishpat"

        entry_idx = 0

        # Check if structure is 2D (siman -> seif) or 3D (part -> siman -> seif)
        if text_data and isinstance(text_data[0], list) and len(text_data[0]) > 0:
            # Check if first element is a list of lists (3D) or list of strings (2D)
            first_item = text_data[0]
            if (
                isinstance(first_item, list)
                and len(first_item) > 0
                and isinstance(first_item[0], list)
            ):
                # 3D structure: part -> siman -> seif
                for part_idx, part_simanim in enumerate(text_data):
                    part_name_local = (
                        part_name if part_idx == 0 else f"Part {part_idx + 1}"
                    )

                    # Get Hebrew part
                    he_part = []
                    if part_idx < len(he_data):
                        he_part = he_data[part_idx]
                        if not isinstance(he_part, list):
                            he_part = [he_part] if he_part else []

                    # Iterate through simanim in this part
                    for siman_idx, siman_seifim in enumerate(part_simanim):
                        if not siman_seifim or (
                            isinstance(siman_seifim, list) and len(siman_seifim) == 0
                        ):
                            continue

                        siman_num = siman_idx + 1

                        # Get Hebrew siman
                        he_siman = []
                        if siman_idx < len(he_part):
                            he_siman = he_part[siman_idx]
                            if not isinstance(he_siman, list):
                                he_siman = [he_siman] if he_siman else []

                        # Iterate through seifim in this siman
                        max_seifim = max(len(siman_seifim), len(he_siman))

                        for seif_idx in range(max_seifim):
                            seif_num = seif_idx + 1

                            # Get English seif
                            seif_text = None
                            if seif_idx < len(siman_seifim):
                                seif_text = siman_seifim[seif_idx]

                            # Get Hebrew seif
                            he_seif = None
                            if seif_idx < len(he_siman):
                                he_seif = he_siman[seif_idx]

                            # Clean text
                            he_seif = self._clean_text(he_seif)
                            seif_text = self._clean_text(seif_text)

                            # Skip if no text
                            if (not he_seif or not he_seif.strip()) and (
                                not seif_text or not seif_text.strip()
                            ):
                                continue

                            # Build ref using metadata-aware method
                            # For Shulchan Arukh: Shulchan Arukh, Orach Chayim.34:1 (Siman:Seif format)
                            sefaria_ref = self._build_ref([siman_num, seif_num])

                            entry = TextEntry(
                                id=f"{self.book.replace('/', '_')}.{entry_idx}",
                                sefaria_ref=sefaria_ref,
                                book=self.book,
                                category=self.categories,
                                text=he_seif if he_seif else "",
                                links=self.links if entry_idx == 0 else [],
                                address_types=self.address_types,
                            )
                            entries.append(entry)
                            entry_idx += 1
            else:
                # 2D structure: siman -> seif
                for siman_idx, siman_seifim in enumerate(text_data):
                    if not siman_seifim or (
                        isinstance(siman_seifim, list) and len(siman_seifim) == 0
                    ):
                        continue

                    siman_num = siman_idx + 1

                    # Get Hebrew siman
                    he_siman = []
                    if siman_idx < len(he_data):
                        he_siman = he_data[siman_idx]
                        if not isinstance(he_siman, list):
                            he_siman = [he_siman] if he_siman else []

                    # Handle case where siman_seifim is a string (single seif)
                    if isinstance(siman_seifim, str):
                        siman_seifim = [siman_seifim]

                    # Iterate through seifim in this siman
                    max_seifim = max(len(siman_seifim), len(he_siman))

                    for seif_idx in range(max_seifim):
                        seif_num = seif_idx + 1

                        # Get English seif
                        seif_text = None
                        if seif_idx < len(siman_seifim):
                            seif_text = siman_seifim[seif_idx]

                        # Get Hebrew seif
                        he_seif = None
                        if seif_idx < len(he_siman):
                            he_seif = he_siman[seif_idx]

                        # Clean text
                        he_seif = self._clean_text(he_seif)
                        seif_text = self._clean_text(seif_text)

                        # Skip if no text
                        if (not he_seif or not he_seif.strip()) and (
                            not seif_text or not seif_text.strip()
                        ):
                            continue

                        # Build ref using metadata-aware method
                        # For Shulchan Arukh: Shulchan Arukh, Orach Chayim.34:1 (Siman:Seif format)
                        sefaria_ref = self._build_ref([siman_num, seif_num])

                        entry = TextEntry(
                            id=f"{self.book.replace('/', '_')}.{entry_idx}",
                            sefaria_ref=sefaria_ref,
                            book=self.book,
                            category=self.categories,
                            text=he_seif if he_seif else "",
                            links=self.links if entry_idx == 0 else [],
                            address_types=self.address_types,
                        )
                        entries.append(entry)
                        entry_idx += 1

        return entries


def normalize(book: str, data_dir: str = "data/raw") -> List[TextEntry]:
    """
    Normalizes a single book into TextEntry items

    Args:
        book: Book name
        data_dir: Raw data directory

    Returns:
        List of TextEntry
    """
    raw = load_raw(book, data_dir)

    # Handle case where raw is a list (shouldn't happen but just in case)
    if isinstance(raw, list):
        print(f"⚠ Warning: {book} is a list, skipping")
        return []

    # Determine book type and use appropriate normalizer
    categories = raw.get("categories", []) if isinstance(raw, dict) else []
    if not categories and isinstance(raw, dict) and "index" in raw:
        categories = raw["index"].get("categories", [])

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

    # Choose appropriate normalizer
    if is_talmud:
        normalizer = TalmudNormalizer(raw, book)
    elif is_mishnah:
        normalizer = MishnahNormalizer(raw, book)
    elif is_shulchan_arukh:
        normalizer = ShulchanArukhNormalizer(raw, book)
    elif is_tanakh:
        normalizer = TanakhNormalizer(raw, book)
    else:
        # Default to TanakhNormalizer for unknown types
        normalizer = TanakhNormalizer(raw, book)

    return normalizer.normalize()


def save_normalized(
    entries: List[TextEntry], book: str, output_dir: str = "data/normalized"
):
    """Saves normalized entries to JSON"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_path = Path(output_dir) / f"{book.replace('/', '_')}.json"

    out = [e.dict() for e in entries]
    output_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    abs_path = output_path.resolve()
    file_size = output_path.stat().st_size
    print(f"✓ Saved: {abs_path}")
    print(f"  Entries: {len(entries)}, File size: {file_size} bytes")


def normalize_all(
    data_dir: str = "data/raw",
    output_dir: str = "data/normalized",
    books: List[str] = None,
):
    """Normalizes all files in the directory"""
    raw_dir = Path(data_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not raw_dir.exists():
        print(f"Directory not found: {raw_dir}")
        return

    json_files = list(raw_dir.glob("*.json"))

    # Filter by books if provided
    if books:
        # Normalize book names for comparison (handle underscores, etc.)
        books_normalized = {book.replace("/", "_") for book in books}
        json_files = [f for f in json_files if f.stem in books_normalized]
        if not json_files:
            print(f"No matching files found for books: {books}")
            return

    print(f"\nNormalizing files from: {raw_dir}")
    print(f"Output directory: {output_path}")
    print(f"Found {len(json_files)} files to process\n")

    for i, json_file in enumerate(json_files, 1):
        book = json_file.stem
        print(f"[{i}/{len(json_files)}] Processing: {book}")
        try:
            entries = normalize(book, str(raw_dir))
            save_normalized(entries, book, str(output_path))
        except Exception as e:
            print(f"✗ Error processing {book}: {e}")

    print(f"\n{'='*60}")
    print(f"Normalization Summary:")
    print(f"  Processed: {len(json_files)} files")
    print(f"  Output directory: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        book = sys.argv[1]
        entries = normalize(book)
        save_normalized(entries, book)
    else:
        normalize_all()
