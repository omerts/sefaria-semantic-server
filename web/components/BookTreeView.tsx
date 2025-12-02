"use client";

import { useState, useCallback } from "react";
import { ChevronDown, ChevronRight, Play, Trash2, Loader2 } from "lucide-react";
import { Category, Book, DeleteResponse } from "@/types";

interface BookTreeViewProps {
  categories: Category[];
  onPipelineRun: (bookTitles: string[]) => Promise<void>;
  onDelete: (book: string) => Promise<DeleteResponse>;
}

interface BookWithPath extends Book {
  fullPath: string; // Full path including category hierarchy
}

export default function BookTreeView({
  categories,
  onPipelineRun,
  onDelete,
}: BookTreeViewProps) {
  const [selectedBooks, setSelectedBooks] = useState<Set<string>>(new Set());
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set()
  );
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Collect all books from categories recursively
  const getAllBooks = useCallback(
    (cats: Category[], parentPath: string = ""): BookWithPath[] => {
      const books: BookWithPath[] = [];
      cats.forEach((category) => {
        const currentPath = parentPath
          ? `${parentPath} > ${category.name}`
          : category.name;
        category.books.forEach((book) => {
          books.push({
            ...book,
            fullPath: currentPath,
          });
        });
        if (category.subcategories.length > 0) {
          books.push(...getAllBooks(category.subcategories, currentPath));
        }
      });
      return books;
    },
    []
  );

  const allBooks = getAllBooks(categories);

  const toggleBookSelection = (bookPath: string) => {
    setSelectedBooks((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(bookPath)) {
        newSet.delete(bookPath);
      } else {
        newSet.add(bookPath);
      }
      return newSet;
    });
  };

  const toggleCategoryExpansion = (categoryPath: string) => {
    setExpandedCategories((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(categoryPath)) {
        newSet.delete(categoryPath);
      } else {
        newSet.add(categoryPath);
      }
      return newSet;
    });
  };

  const selectAll = () => {
    if (selectedBooks.size === allBooks.length) {
      setSelectedBooks(new Set());
    } else {
      setSelectedBooks(new Set(allBooks.map((book) => book.path)));
    }
  };

  const handleBulkPipeline = async () => {
    if (selectedBooks.size === 0) return;

    try {
      setIsPipelineRunning(true);
      const selectedBookTitles = Array.from(selectedBooks).map(
        (bookPath) =>
          allBooks.find((book) => book.path === bookPath)?.title || bookPath
      );

      await onPipelineRun(selectedBookTitles as string[]);

      alert(`Pipeline הופעל עבור ${selectedBooks.size} ספרים`);
      setSelectedBooks(new Set());
    } catch (err) {
      console.error("Pipeline error:", err);
      alert("שגיאה בהרצת Pipeline");
    } finally {
      setIsPipelineRunning(false);
    }
  };

  const handleBulkDelete = async () => {
    if (selectedBooks.size === 0) return;

    if (
      !confirm(
        `האם אתה בטוח שברצונך למחוק ${selectedBooks.size} ספרים? פעולה זו לא ניתנת לביטול.`
      )
    ) {
      return;
    }

    try {
      setIsDeleting(true);
      const selectedBookTitles = Array.from(selectedBooks).map(
        (bookPath) =>
          allBooks.find((book) => book.path === bookPath)?.title || bookPath
      );

      for (const bookTitle of selectedBookTitles) {
        await onDelete(bookTitle);
      }

      alert(`נמחקו ${selectedBooks.size} ספרים`);
      setSelectedBooks(new Set());
    } catch (err) {
      console.error("Delete error:", err);
      alert("שגיאה במחיקה");
    } finally {
      setIsDeleting(false);
    }
  };

  const renderCategory = (
    category: Category,
    level: number = 0,
    parentPath: string = ""
  ) => {
    const categoryPath = parentPath
      ? `${parentPath} > ${category.name}`
      : category.name;
    const isExpanded = expandedCategories.has(categoryPath);
    const hasContent =
      category.books.length > 0 || category.subcategories.length > 0;

    if (!hasContent) return null;

    const indentStyle = { marginRight: `${level * 1}rem` };

    return (
      <div key={categoryPath} style={indentStyle}>
        <div className="flex items-center py-2 hover:bg-gray-50 dark:hover:bg-gray-800 rounded">
          <button
            onClick={() => toggleCategoryExpansion(categoryPath)}
            className="flex items-center gap-2 flex-1 text-right"
          >
            {isExpanded ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
            <span className="font-semibold text-gray-900 dark:text-white">
              {category.name}
            </span>
            {category.he_name && (
              <span className="text-sm text-gray-600 dark:text-gray-400">
                ({category.he_name})
              </span>
            )}
            {category.books.length > 0 && (
              <span className="text-sm text-gray-500 dark:text-gray-400">
                ({category.books.length} ספרים)
              </span>
            )}
          </button>
        </div>

        {isExpanded && (
          <div className="mr-4">
            {category.books.map((book) => (
              <div
                key={book.path}
                className="flex items-center py-2 px-2 hover:bg-gray-50 dark:hover:bg-gray-800 rounded"
              >
                <input
                  type="checkbox"
                  checked={selectedBooks.has(book.path)}
                  onChange={() => toggleBookSelection(book.path)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600 ml-3"
                />
                <div className="flex-1 text-right">
                  <div className="font-medium text-gray-900 dark:text-white">
                    {book.title}
                  </div>
                  {book.he_title && (
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {book.he_title}
                    </div>
                  )}
                  <div className="text-xs text-gray-500 dark:text-gray-500">
                    {book.path}
                  </div>
                </div>
              </div>
            ))}

            {category.subcategories.map((subcategory) =>
              renderCategory(subcategory, level + 1, categoryPath)
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Action Bar */}
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <button
              onClick={selectAll}
              className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
            >
              {selectedBooks.size === allBooks.length ? "בטל בחירה" : "בחר הכל"}
            </button>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              נבחרו: {selectedBooks.size} מתוך {allBooks.length}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleBulkPipeline}
              disabled={
                selectedBooks.size === 0 || isPipelineRunning || isDeleting
              }
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed text-white rounded-md transition-colors text-sm"
            >
              {isPipelineRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  מריץ...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  הרץ Pipeline ({selectedBooks.size})
                </>
              )}
            </button>

            <button
              onClick={handleBulkDelete}
              disabled={
                selectedBooks.size === 0 || isPipelineRunning || isDeleting
              }
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-400 disabled:cursor-not-allowed text-white rounded-md transition-colors text-sm"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  מוחק...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4" />
                  מחק ({selectedBooks.size})
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Tree View */}
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
          עץ הספרים
        </h3>
        <div className="space-y-1">
          {categories.map((category) => renderCategory(category))}
        </div>
      </div>
    </div>
  );
}
