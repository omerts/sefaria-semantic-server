"""
Download data from Sefaria API
"""

import requests
import json
from pathlib import Path
import time
from typing import List, Optional


EXPORT_URL = "https://www.sefaria.org/api/bulktext"
BASE_URL = "https://www.sefaria.org/api"
V3_BASE_URL = "https://www.sefaria.org/api/v3"


def download_book(
    book: str,
    output_dir: str = "data/raw",
    version: str = "hebrew",
    max_retries: int = 3,
) -> bool:
    """
    Downloads a single book from Sefaria using v3 API
    Returns Hebrew text only, without nikud (vowel points)

    Args:
        book: Book name (e.g., "Berakhot", "Genesis")
        output_dir: Output directory
        version: Language/version to download (default: "hebrew")
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        True if successful, False otherwise
    """
    # Use v3 API with specified version and text_only format
    url = f"{V3_BASE_URL}/texts/{book}"
    params = {
        "version": version,  # Get text in specified language/version
        "return_format": "text_only",  # Plain text without HTML, footnotes, etc.
    }

    headers = {"accept": "application/json"}

    # Retry logic with exponential backoff for large books
    for attempt in range(max_retries):
        try:
            print(f"Downloading: {book}...")
            if attempt > 0:
                print(f"  Retry attempt {attempt + 1}/{max_retries}")
            print(f"  URL: {url}")
            print(f"  Params: {params}")

            # Increase timeout for large books (120 seconds)
            # For very large books like Genesis, may need even more time
            timeout = 120 if attempt == 0 else 180

            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            r.raise_for_status()

            # Parse JSON response from v3 API
            data = r.json()

            # Extract text and metadata for mapping
            # v3 API returns data with structure containing both text and metadata
            text_data = data
            if "versions" in data and len(data["versions"]) > 0:
                # Get the version text
                version_data = data["versions"][0]
                if "text" in version_data:
                    text_data = version_data["text"]
                else:
                    text_data = version_data

            # Build output structure with text and important mapping metadata
            output_data = {
                "text": text_data,
                "he": text_data,  # For compatibility with existing normalizers
            }

            # Extract important mapping metadata from the response
            # These fields are crucial for mapping sections correctly
            if isinstance(data, dict):
                # Core mapping fields
                if "sectionNames" in data:
                    output_data["sectionNames"] = data["sectionNames"]
                if "addressTypes" in data:
                    output_data["addressTypes"] = data["addressTypes"]
                if "textDepth" in data:
                    output_data["textDepth"] = data["textDepth"]
                if "lengths" in data:
                    output_data["lengths"] = data["lengths"]
                if "length" in data:
                    output_data["length"] = data["length"]

                # Reference fields for mapping
                if "sectionRef" in data:
                    output_data["sectionRef"] = data["sectionRef"]
                if "firstAvailableSectionRef" in data:
                    output_data["firstAvailableSectionRef"] = data[
                        "firstAvailableSectionRef"
                    ]
                if "heSectionRef" in data:
                    output_data["heSectionRef"] = data["heSectionRef"]

                # Order and structure
                if "order" in data:
                    output_data["order"] = data["order"]
                if "alts" in data:
                    output_data["alts"] = data["alts"]

                # Book metadata
                if "title" in data:
                    output_data["title"] = data["title"]
                if "heTitle" in data:
                    output_data["heTitle"] = data["heTitle"]
                if "book" in data:
                    output_data["book"] = data["book"]
                if "categories" in data:
                    output_data["categories"] = data["categories"]

                # Index metadata (if exists)
                if "index" in data and isinstance(data["index"], dict):
                    output_data["index"] = data["index"]

            Path(output_dir).mkdir(exist_ok=True, parents=True)
            output_path = Path(output_dir) / f"{book.replace('/', '_')}.json"

            # Save as JSON with text and mapping metadata
            output_path.write_text(
                json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Show absolute path
            abs_path = output_path.resolve()
            file_size = output_path.stat().st_size
            print(f"✓ Saved: {abs_path}")
            print(
                f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)"
            )
            return True

        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                wait_time = (2**attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                print(f"  ⚠ Timeout after {timeout}s. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(
                    f"✗ Error downloading {book}: Timeout after {max_retries} attempts"
                )
                return False
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (2**attempt) * 2  # Shorter backoff for other errors
                print(f"  ⚠ Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"✗ Error downloading {book}: {e}")
                return False
        except Exception as e:
            print(f"✗ Error processing {book}: {e}")
            import traceback

            traceback.print_exc()
            return False

    # Should not reach here, but just in case
    return False


def get_all_books() -> List[str]:
    """
    Returns a list of all available books from Sefaria
    """
    try:
        r = requests.get(f"{BASE_URL}/index", timeout=30)
        r.raise_for_status()
        data = r.json()

        books = []

        def extract_books(node, prefix=""):
            if isinstance(node, dict):
                if "title" in node:
                    full_title = (
                        f"{prefix}.{node['title']}" if prefix else node["title"]
                    )
                    if node.get("isBook", False):
                        books.append(full_title)
                    if "contents" in node:
                        for child in node["contents"]:
                            extract_books(child, full_title)

        for category in data.get("categories", []):
            extract_books(category)

        return books
    except Exception as e:
        print(f"Error getting book list: {e}")
        return []


def download_all(
    books: List[str],
    delay: float = 1.0,
    output_dir: str = "data/raw",
    version: str = "hebrew",
):
    """
    Downloads a list of books with delay between requests

    Args:
        books: List of book names
        delay: Delay in seconds between requests (to avoid overloading the server)
        output_dir: Output directory
        version: Language/version to download (default: "hebrew")
    """
    output_path = Path(output_dir).resolve()
    print(f"\nDownloading to: {output_path}")
    print(f"Language/Version: {version}")
    print(f"Total books to download: {len(books)}\n")

    success_count = 0
    for i, book in enumerate(books, 1):
        print(f"[{i}/{len(books)}] ", end="")
        if download_book(book, output_dir, version):
            success_count += 1

        if i < len(books):
            time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"  Successfully downloaded: {success_count}/{len(books)} books")
    print(f"  Files saved to: {output_path}")
    print(f"{'='*60}")


# Default books to download when running as part of pipeline
DEFAULT_BOOKS = [
    "Berakhot",
    "Genesis",
    "Mishnah Berakhot",
    "Rambam Mishneh Torah, Teshuva",
    "Shulchan Arukh, Orach Chayim",
]


def run_download(
    output_dir: str = "data/raw", books: List[str] = None, version: str = "hebrew"
):
    """
    Downloads books - used by pipeline

    Args:
        output_dir: Output directory
        books: List of books to download (if None, uses DEFAULT_BOOKS)
        version: Language/version to download (default: "hebrew")
    """
    if books is None:
        books = DEFAULT_BOOKS

    print(f"Downloading {len(books)} books:")
    for book in books:
        print(f"  - {book}")
    print(f"Language/Version: {version}")
    print()

    download_all(books, output_dir=output_dir, version=version)


if __name__ == "__main__":
    # Interactive mode when run directly
    print("Selecting books to download...")
    print("Option 1: Sample books")
    print("Option 2: All books (very long!)")

    choice = input("Enter choice (1/2): ").strip()

    if choice == "2":
        all_books = get_all_books()
        print(f"Found {len(all_books)} books")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == "y":
            download_all(all_books)
    else:
        download_all(DEFAULT_BOOKS)
