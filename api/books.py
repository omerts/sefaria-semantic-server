"""
API endpoints for books management
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

router = APIRouter()


class BookInfo(BaseModel):
    """Book information"""

    title: str
    he_title: Optional[str] = None
    categories: List[str] = []
    path: str  # Full path like "Tanakh.Torah.Genesis"


class CategoryInfo(BaseModel):
    """Category information with books"""

    name: str
    he_name: Optional[str] = None
    books: List[BookInfo] = []
    subcategories: List["CategoryInfo"] = []


CategoryInfo.model_rebuild()


class BooksResponse(BaseModel):
    """Books response organized by categories"""

    categories: List[CategoryInfo]
    total_books: int


def get_books_by_categories() -> Dict:
    """
    Gets all books from Sefaria API and organizes them by categories

    Returns:
        Dictionary with category structure and books

    Raises:
        Exception: If failed to fetch or parse books
    """
    import requests
    import traceback

    try:
        BASE_URL = "https://www.sefaria.org/api"
        r = requests.get(f"{BASE_URL}/index", timeout=30)
        r.raise_for_status()
        raw_data = r.json()

        # Handle both dict and list responses from Sefaria API
        # According to download_sefaria.py, it should be a dict with 'categories' key
        # But sometimes the API might return a list directly
        if isinstance(raw_data, list):
            # If it's a list, treat it as categories
            data = {"categories": raw_data}
        elif isinstance(raw_data, dict):
            # If it's a dict, use it as is (should have 'categories' key)
            data = raw_data
            if "categories" not in data:
                # If 'categories' key doesn't exist, try to use the dict itself
                # or check if there's another structure
                print(
                    f"Warning: API response dict missing 'categories' key. Keys: {list(data.keys())}"
                )
                # Try to find categories in the data
                if len(data) == 1 and isinstance(list(data.values())[0], list):
                    # Maybe the data is wrapped differently
                    data = {"categories": list(data.values())[0]}
                else:
                    # Last resort: wrap the whole dict as a single category
                    data = {"categories": [raw_data]}
        else:
            raise ValueError(f"Unexpected API response type: {type(raw_data)}")

        # Ensure we have categories
        if "categories" not in data or not isinstance(data["categories"], list):
            raise ValueError(
                f"Invalid data structure: missing or invalid 'categories' field"
            )

    except Exception as e:
        error_msg = (
            f"Failed to fetch from Sefaria API: {str(e)}\n{traceback.format_exc()}"
        )
        print(error_msg)
        raise Exception(error_msg)

    def build_category_tree(node, parent_path=""):
        """Recursively build category tree"""
        if not isinstance(node, dict):
            return None

        # Handle both 'title' (for nested categories) and 'category' (for top-level)
        category_name = node.get("title") or node.get("category", "")
        if not category_name:
            return None

        # Handle both 'heTitle' (for nested) and 'heCategory' (for top-level)
        he_category_name = node.get("heTitle") or node.get("heCategory")
        full_path = f"{parent_path}.{category_name}" if parent_path else category_name

        category_info = {
            "name": category_name,
            "he_name": he_category_name,
            "path": full_path,
            "books": [],
            "subcategories": [],
        }

        # Process contents
        if "contents" in node:
            for child in node.get("contents", []):
                if not isinstance(child, dict):
                    continue

                # A book is identified by:
                # 1. Having a 'title' field (not just 'category')
                # 2. NOT having 'contents' field, or having empty 'contents'
                # 3. The 'isBook' flag might be set, but it's not reliable
                has_title = bool(child.get("title"))
                has_contents = "contents" in child
                contents_empty = not child.get("contents", [])
                is_book_flag = child.get("isBook", False)

                # Check if this is a book
                is_book = (
                    has_title
                    and (not has_contents or contents_empty or is_book_flag)
                    # Exclude top-level categories that have titles but are actually categories
                    and child.get("category") != child.get("title")
                )

                if is_book:
                    # This is a book
                    book_title = child.get("title", "")
                    if not book_title:
                        continue

                    book_he_title = child.get("heTitle")
                    book_path = f"{full_path}.{book_title}" if full_path else book_title

                    # Extract categories from path
                    categories = [c.strip() for c in full_path.split(".") if c.strip()]

                    book_info = {
                        "title": book_title,
                        "he_title": book_he_title,
                        "categories": categories,
                        "path": book_path,
                    }
                    category_info["books"].append(book_info)
                else:
                    # This is a subcategory - recursively process it
                    subcategory = build_category_tree(child, full_path)
                    if subcategory:
                        category_info["subcategories"].append(subcategory)

        # Return category even if it has no books or subcategories (might be empty)
        return category_info

    categories = []
    for category_node in data.get("categories", []):
        category_info = build_category_tree(category_node)
        # Include category even if it has no books (it might have subcategories with books)
        if category_info:
            categories.append(category_info)

    # Debug: print some info
    total_books_found = sum(
        len(cat.get("books", []))
        + sum(len(sub.get("books", [])) for sub in cat.get("subcategories", []))
        for cat in categories
    )
    print(
        f"Debug: Found {len(categories)} top-level categories, {total_books_found} total books"
    )

    return {"categories": categories}


@router.get("/books", response_model=BooksResponse)
async def get_books():
    """
    Get all books organized by categories

    Returns books from Sefaria API organized in a hierarchical category structure
    """
    try:
        result = get_books_by_categories()
        categories = result.get("categories", [])

        # Count total books recursively
        def count_books(cat_list):
            total = 0
            for cat in cat_list:
                total += len(cat.get("books", []))
                total += count_books(cat.get("subcategories", []))
            return total

        total_books = count_books(categories)

        # Convert to Pydantic models
        def convert_category(cat_dict):
            try:
                books = [BookInfo(**book) for book in cat_dict.get("books", [])]
                subcategories = [
                    convert_category(sub) for sub in cat_dict.get("subcategories", [])
                ]
                return CategoryInfo(
                    name=cat_dict.get("name", ""),
                    he_name=cat_dict.get("he_name"),
                    books=books,
                    subcategories=subcategories,
                )
            except Exception as e:
                import traceback

                print(
                    f"Error converting category {cat_dict.get('name', 'unknown')}: {e}"
                )
                traceback.print_exc()
                # Return a minimal valid category
                return CategoryInfo(
                    name=cat_dict.get("name", "Unknown"),
                    he_name=cat_dict.get("he_name"),
                    books=[],
                    subcategories=[],
                )

        category_models = [convert_category(cat) for cat in categories]

        return BooksResponse(categories=category_models, total_books=total_books)
    except Exception as e:
        import traceback

        error_detail = f"Failed to get books: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Failed to get books: {str(e)}")


@router.get("/books/flat")
async def get_books_flat():
    """
    Get all books as a flat list

    Returns a simple list of all books with their categories
    """
    try:
        result = get_books_by_categories()
        categories = result.get("categories", [])

        books_flat = []

        def extract_books(cat_list):
            for cat in cat_list:
                books_flat.extend(cat.get("books", []))
                extract_books(cat.get("subcategories", []))

        extract_books(categories)

        return {
            "books": [BookInfo(**book) for book in books_flat],
            "total": len(books_flat),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get books: {str(e)}")
