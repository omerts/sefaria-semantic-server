"use client";

import { useState } from "react";
import { Trash2, Loader2, AlertTriangle } from "lucide-react";
import { useEmbeddingModel } from "@/contexts/EmbeddingModelContext";

interface DeleteControlsProps {
  onDelete: (book?: string) => Promise<void>;
  selectedBook?: string;
}

export default function DeleteControls({
  onDelete,
  selectedBook,
}: DeleteControlsProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const { selectedModel } = useEmbeddingModel();

  const handleDelete = async (book?: string) => {
    if (!showConfirm) {
      setShowConfirm(true);
      return;
    }

    try {
      setIsDeleting(true);
      await onDelete(book);
      setShowConfirm(false);
    } catch (err) {
      console.error("Delete error:", err);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleCancel = () => {
    setShowConfirm(false);
  };

  if (showConfirm) {
    return (
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200 mb-2">
              {selectedBook
                ? `האם אתה בטוח שברצונך למחוק את כל ה-chunks של הספר "${selectedBook}"?`
                : `האם אתה בטוח שברצונך למחוק את כל ה-chunks מהקולקציה "${selectedModel}"?`}
            </p>
            <p className="text-xs text-yellow-700 dark:text-yellow-300 mb-3">
              פעולה זו אינה ניתנת לביטול!
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => handleDelete(selectedBook)}
                disabled={isDeleting}
                className="px-3 py-1.5 bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white rounded-md transition-colors text-sm flex items-center gap-1"
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    מוחק...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" />
                    מחק
                  </>
                )}
              </button>
              <button
                onClick={handleCancel}
                disabled={isDeleting}
                className="px-3 py-1.5 bg-gray-300 hover:bg-gray-400 disabled:bg-gray-200 text-gray-800 rounded-md transition-colors text-sm"
              >
                ביטול
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={() => handleDelete(selectedBook)}
      disabled={isDeleting}
      className={`${
        selectedBook
          ? "px-2 py-1 bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white rounded-md transition-colors flex items-center gap-1 text-xs"
          : "px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white rounded-lg transition-colors flex items-center gap-2 text-sm"
      }`}
      title={
        selectedBook
          ? `מחק את הספר "${selectedBook}"`
          : "מחק את כל ה-chunks מהקולקציה"
      }
    >
      {isDeleting ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin" />
          {selectedBook ? "" : "מוחק..."}
        </>
      ) : (
        <>
          <Trash2 className="w-4 h-4" />
          {selectedBook ? "" : "מחק הכל"}
        </>
      )}
    </button>
  );
}
