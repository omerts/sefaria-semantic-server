.PHONY: install setup download normalize chunk embed index train-all run-api help

help:
	@echo "Torah Source Finder - Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Full setup (venv + install)"
	@echo "  make download    - Download data from Sefaria"
	@echo "  make normalize   - Normalize downloaded data"
	@echo "  make chunk       - Create chunks from normalized data"
	@echo "  make embed       - Generate embeddings for chunks"
	@echo "  make index       - Index chunks in Qdrant"
	@echo "  make train-data  - Build training data pairs"
	@echo "  make train-all   - Train both retriever and reranker"
	@echo "  make run-api     - Run FastAPI server"
	@echo "  make pipeline    - Run full pipeline"

install:
	pip install -r requirements.txt

setup:
	bash setup.sh

download:
	python -m ingestion.download_sefaria

normalize:
	python -m ingestion.normalize

chunk:
	python -m preprocess.chunker

embed:
	python -m preprocess.embedder

index:
	python -m preprocess.indexer

train-data:
	python -m training.build_training_data

train-retriever:
	python -m training.train_retriever

train-reranker:
	python -m training.train_reranker

train-all: train-data train-retriever train-reranker

run-api:
	uvicorn api.main:app --reload

pipeline:
	python run_pipeline.py

