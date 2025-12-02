#!/usr/bin/env python3
"""
Search test script
"""
import requests
import json
import sys


def test_search(query: str, base_url: str = "http://localhost:8000"):
    """Tests the search endpoint"""
    url = f"{base_url}/api/search"

    payload = {"query": query, "limit": 5}

    print(f"Sending query: {query}")
    print(f"URL: {url}")
    print()

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()

        print(f"Found {data.get('total', 0)} results:")
        print("=" * 60)

        for i, result in enumerate(data.get("results", []), 1):
            print(f"\n{i}. {result.get('book')} - {result.get('sefaria_ref')}")
            print(f"   Score: {result.get('score', 0):.4f}")
            print(f"   Text: {result.get('text', '')[:100]}...")

        return True

    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server")
        print("  Make sure API is running: uvicorn api.main:app --reload")
        return False
    except requests.exceptions.Timeout:
        print("✗ Timeout")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_health(base_url: str = "http://localhost:8000"):
    """Tests the health endpoint"""
    url = f"{base_url}/api/health"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()
        print("Health Check:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return True

    except Exception as e:
        print(f"✗ Error in health check: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Torah Source Finder - Test")
    print("=" * 60)
    print()

    # Test health
    print("1. Testing health endpoint...")
    if not test_health():
        sys.exit(1)

    print()
    print("2. Testing search endpoint...")

    # Example queries
    queries = ["כדי להרחיק את האדם מן "]

    if len(sys.argv) > 1:
        queries = [sys.argv[1]]

    for query in queries:
        print()
        if not test_search(query):
            sys.exit(1)

    print()
    print("=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
