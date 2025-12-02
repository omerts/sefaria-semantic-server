# Torah Source Finder

A Torah source search system based on Sefaria with embeddings and re-ranking.

## Project Structure

```
torah-source-finder/
├── ingestion/          # Data collection from Sefaria
│   ├── download_sefaria.py
│   ├── normalize.py
│   └── schema.py
├── preprocess/         # Text processing, chunking, embeddings
│   ├── chunker.py
│   ├── embedder.py
│   ├── indexer.py
│   └── summarizer.py
├── models/             # Retriever and reranker models
│   ├── retriever.py
│   └── reranker.py
├── api/                # FastAPI endpoints
│   ├── main.py
│   ├── search.py
│   └── dependencies.py
├── training/           # Model training
│   ├── build_training_data.py
│   ├── train_retriever.py
│   └── train_reranker.py
├── data/               # Raw and processed data
│   ├── raw/
│   ├── normalized/
│   └── chunks/
├── run_pipeline.py     # Script to run the full pipeline
└── requirements.txt
```

## Prerequisites

1. **Python 3.8+**
2. **Qdrant** - Vector Database
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
   Or local installation: https://qdrant.tech/documentation/quick-start/

## Installation

### Option 1: Using uv (Recommended)

```bash
# Install dependencies using uv
uv pip install -r requirements.txt --native-tls

# Or use the setup script
bash setup.sh
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage - Full Pipeline

### Option 1: Automatic pipeline execution

**Using venv:**

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python run_pipeline.py
```

**Using uv run:**

```bash
# Note: uv run creates its own environment. If you have SSL issues, use venv instead.
uv run run_pipeline.py
```

**If you have SSL certificate issues, use venv with DISABLE_SSL_VERIFY:**

```bash
# Activate venv and disable SSL verification
source venv/bin/activate
export DISABLE_SSL_VERIFY=1
python run_pipeline.py
```

### Option 2: Manual step-by-step execution

#### Step 1: Download data from Sefaria

```bash
python -m ingestion.download_sefaria
```

This will download sample books. To update, edit the file and replace `sample_books`.

#### Step 2: Normalize and process data

```bash
python -m ingestion.normalize
```

This will clean the text, remove nikud (vowel marks), and create a uniform structure.

#### Step 3: Create chunks

```bash
python -m preprocess.chunker
```

Divides text into chunks according to book type (Tanakh, Mishnah, Halacha, etc.).

#### Step 4: Generate embeddings

```bash
python -m preprocess.embedder
```

Creates embeddings for each chunk using the BGE-M3 model (multilingual).

#### Step 5: Index in Qdrant

```bash
python -m preprocess.indexer
```

Uploads all chunks and embeddings to Qdrant.

## Running the API

```bash
uvicorn api.main:app --reload
```

The API will be available at: http://localhost:8000

### Automatic Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### POST `/api/search`

Search for Torah sources.

**Request:**

```json
{
  "query": "What is the law regarding blessing on new fruit",
  "limit": 10,
  "score_threshold": 0.0,
  "book": null,
  "category": null
}
```

**Response:**

```json
{
  "results": [
    {
      "sefaria_ref": "Berakhot.1.1",
      "book": "Berakhot",
      "category": ["Mishnah"],
      "text": "מאימתי קורין את שמע בערבין",
      "score": 0.95,
      "position": 0,
      "chunk_type": "mishnah"
    }
  ],
  "total": 1,
  "query": "What is the law regarding blessing on new fruit"
}
```

### GET `/api/health`

System health check.

## Model Training (Optional)

### Step 1: Build training pairs

```bash
python -m training.build_training_data
```

This creates positive and negative pairs from Sefaria links.

### Step 2: Train Retriever

```bash
python -m training.train_retriever
```

Improves embeddings for better search.

### Step 3: Train Re-Ranker

```bash
python -m training.train_reranker
```

Improves result ranking.

After training, set environment variables:

```bash
export RETRIEVER_MODEL_PATH=models/retriever
export RERANKER_MODEL_PATH=models/reranker
```

## Environment Settings (Optional)

Create a `.env` file:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=sefaria_chunks
RETRIEVER_MODEL_PATH=models/retriever
RERANKER_MODEL_PATH=models/reranker
```

## Notes

- Base models (BGE-M3, CrossEncoder) support Hebrew but are not specifically adapted for Torah text
- Model training significantly improves accuracy
- Qdrant must be running before indexing and search
- Downloading all Sefaria books takes a long time - recommended to start with specific books

## Common Issues

**Qdrant connection failed:**

- Make sure Qdrant is running: `docker ps`
- Check the port: `curl http://localhost:6333/health`

**Models not loading:**

- Make sure you installed all requirements
- For FlagEmbedding, you may need: `pip install FlagEmbedding --upgrade`

**SSL Certificate errors when downloading models:**

- **Option 1: Use a custom certificate file:**

  ```bash
  source venv/bin/activate
  export SSL_CERT_FILE=my-cert.pem
  python run_pipeline.py
  ```

- **Option 2: Disable SSL verification (use with caution):**

  ```bash
  source venv/bin/activate
  export DISABLE_SSL_VERIFY=1
  python run_pipeline.py
  ```

  This sets `REQUESTS_CA_BUNDLE=""` and disables SSL verification.

- **Note:** `uv run` may have SSL issues when downloading packages. Use `venv` instead if you encounter SSL errors.
- **Note:** Only disable SSL verification if you trust your network connection

**SSL Certificate errors with uv:**

- If `uv` itself has SSL issues when downloading packages, use `venv` instead:
  ```bash
  source venv/bin/activate
  python run_pipeline.py
  ```
- Or install packages manually with `uv pip install` in your venv:
  ```bash
  source venv/bin/activate
  uv pip install -r requirements.txt --native-tls
  ```

**Insufficient memory:**

- Reduce `batch_size` in embedder and indexer
- Use GPU if available (add `device=0` in embedder)
