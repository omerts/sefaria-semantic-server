"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

export type EmbeddingModel = "BAAI/bge-m3" | "MPA/sambert" | "imvladikon/sentence-transformers-alephbert";

interface EmbeddingModelContextType {
  selectedModel: EmbeddingModel;
  setSelectedModel: (model: EmbeddingModel) => void;
  availableModels: EmbeddingModel[];
}

const EmbeddingModelContext = createContext<EmbeddingModelContextType | undefined>(undefined);

const DEFAULT_MODEL: EmbeddingModel = "BAAI/bge-m3";
const AVAILABLE_MODELS: EmbeddingModel[] = [
  "BAAI/bge-m3", 
  "MPA/sambert", 
  "imvladikon/sentence-transformers-alephbert"
];

export function EmbeddingModelProvider({ children }: { children: React.ReactNode }) {
  const [selectedModel, setSelectedModelState] = useState<EmbeddingModel>(DEFAULT_MODEL);

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("embeddingModel") as EmbeddingModel | null;
    if (saved && AVAILABLE_MODELS.includes(saved)) {
      setSelectedModelState(saved);
    }
  }, []);

  // Save to localStorage when changed
  const setSelectedModel = (model: EmbeddingModel) => {
    setSelectedModelState(model);
    localStorage.setItem("embeddingModel", model);
  };

  return (
    <EmbeddingModelContext.Provider
      value={{
        selectedModel,
        setSelectedModel,
        availableModels: AVAILABLE_MODELS,
      }}
    >
      {children}
    </EmbeddingModelContext.Provider>
  );
}

export function useEmbeddingModel() {
  const context = useContext(EmbeddingModelContext);
  if (context === undefined) {
    throw new Error("useEmbeddingModel must be used within an EmbeddingModelProvider");
  }
  return context;
}

