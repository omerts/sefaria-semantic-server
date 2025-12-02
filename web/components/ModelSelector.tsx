"use client";

import { useEmbeddingModel } from "@/contexts/EmbeddingModelContext";

export default function ModelSelector() {
  const { selectedModel, setSelectedModel, availableModels } = useEmbeddingModel();

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 mb-4">
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          מודל Embedding:
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value as any)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
        >
          {availableModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          (המודל נשמר אוטומטית)
        </span>
      </div>
    </div>
  );
}

