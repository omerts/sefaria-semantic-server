"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import BookTreeView from "@/components/BookTreeView";
import SearchBar from "@/components/SearchBar";
import ModelSelector from "@/components/ModelSelector";
import DeleteControls from "@/components/DeleteControls";
import { Book, Category, DeleteResponse } from "@/types";
import { useEmbeddingModel } from "@/contexts/EmbeddingModelContext";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:9000";

export default function Home() {
  const [categories, setCategories] = useState<Category[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const { selectedModel } = useEmbeddingModel();

  useEffect(() => {
    fetchBooks();
  }, []);

  const fetchBooks = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE}/api/books`);
      setCategories(response.data.categories || []);
      setError(null);
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.detail || err.message || "Failed to load books";
      setError(errorMessage);
      console.error("Error fetching books:", err);
      console.error("Error response:", err.response?.data);
    } finally {
      setLoading(false);
    }
  };

  const handlePipelineRun = async (bookTitles: string[]) => {
    try {
      const response = await axios.post(`${API_BASE}/api/pipeline`, {
        books: bookTitles,
        language: "hebrew",
        embedding_model: selectedModel,
      });
      alert(
        `Pipeline started for ${bookTitles.join(
          ", "
        )}. Check console for progress.`
      );
      return response.data;
    } catch (err: any) {
      alert(`Failed to start pipeline: ${err.message}`);
      throw err;
    }
  };

  const handleSearch = async (query: string, bookPath?: string) => {
    try {
      const response = await axios.post(`${API_BASE}/api/search`, {
        query,
        limit: 10,
        book: bookPath,
        embedding_model: selectedModel,
      });
      return response.data;
    } catch (err: any) {
      alert(`Search failed: ${err.message}`);
      throw err;
    }
  };

  const handleDelete = async (book?: string) => {
    try {
      const params = new URLSearchParams();
      if (book) {
        params.append("book", book);
      }
      params.append("embedding_model", selectedModel);

      const response = await axios.delete<DeleteResponse>(
        `${API_BASE}/api/delete?${params.toString()}`
      );
      alert(
        response.data.message ||
          `Deleted ${response.data.deleted_count} chunks${
            book ? ` for book "${book}"` : ""
          }`
      );
      return response.data;
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.detail || err.message || "Failed to delete";
      alert(`Delete failed: ${errorMessage}`);
      throw err;
    }
  };

  const handleDeleteForControls = async (book?: string) => {
    await handleDelete(book);
  };

  const handleDeleteForTree = async (book: string) => {
    return await handleDelete(book);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">טוען ספרים...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-red-500 text-xl">שגיאה: {error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Torah Source Finder
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            ניהול וחיפוש ספרי תורה ממאגר ספריא
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <ModelSelector />
        <div className="mb-6">
          <SearchBar
            onSearch={handleSearch}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
          />
        </div>
        <div className="mb-6">
          <DeleteControls onDelete={handleDeleteForControls} />
        </div>

        <div className="mt-8">
          <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
            עץ הספרים
          </h2>
          {categories.length === 0 ? (
            <div className="text-gray-500 dark:text-gray-400">
              לא נמצאו ספרים
            </div>
          ) : (
            <BookTreeView
              categories={categories}
              onPipelineRun={handlePipelineRun}
              onDelete={handleDeleteForTree}
            />
          )}
        </div>
      </main>
    </div>
  );
}
