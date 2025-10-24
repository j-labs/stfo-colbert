# stfo-colbert

> Straightforward ColBERT indexing and serving (if you need a development ColBERT server)

## Design Goals

- **Straightforward**: Single-command usage via CLI (stfo is for "straightforward")
- **Minimal**: Readable, functional code with minimal default dependencies
- **Simple**: One HTTP endpoint only: `GET /search`
- **For development usage**: Suitable for anyone who needs an adhoc sematic search server

## When to Use

Use **stfo-colbert** when you:
- Have a small-to-medium collection and want a simple way to build a ColBERT-style index (via PyLate) and query it over HTTP
- Prefer a one-shot CLI to index and serve, without additional orchestration

## Installation

### From PyPI

```bash
pip install stfo-colbert
```

### From source (development)

```bash
git clone <repository-url>
cd stfo_colbert
pip install -e .
```

## Quickstart

### 1. Install the package

```bash
pip install stfo-colbert
```

### 2. Run the CLI (index and serve)

```bash
stfo-colbert \
  --dataset-path /path/to/dataset.txt
```

### 3. Query the API

```bash
curl "http://127.0.0.1:8889/search?query=hello&k=2"
```

### 4. Example response

```json
{
  "query": "hello",
  "topk": [
    {
      "pid": "1",
      "rank": 0,
      "score": 0.92,
      "text": "Hello world! This is a sample document.",
      "prob": 0.25
    },
    {
      "pid": "2",
      "rank": 1,
      "score": 0.87,
      "text": "A friendly hello from another document.",
      "prob": 0.20
    }
  ]
}
```

## CLI Reference

```bash
stfo-colbert [options]
```

### Options

| Option | Description | Default                                    |
|--------|-------------|--------------------------------------------|
| `--port` | Port to serve on | `8889`                                     |
| `--model-name` | Hugging Face model id/name | `mixedbread-ai/mxbai-edge-colbert-v0-17m`  |
| `--index-path` | Path to existing PyLate index directory | (mutually exclusive with `--dataset-path`) |
| `--dataset-path` | Path to dataset for index creation (file or directory) | -                                          |

## Usage Patterns

**Serve an existing index:**
```bash
stfo-colbert --index-path ./experiments/my_index --port 8889
```

**Build from a delimited TXT, then serve:**
```bash
stfo-colbert --dataset-path ./data/my_corpus.txt --port 8889
```

**Build from a directory of docs, then serve:**
```bash
stfo-colbert --dataset-path ./docs_dir --port 8889
```

## Dataset Formats

### 1. Delimited text file (default)

A plain text file where each document is separated by the delimiter: `\n\n--------\n\n`

**Example:**
```
Document one text

--------

Document two text
```

> **Note:** Any occurrences of the delimiter inside documents are removed during preprocessing to avoid boundary confusion.

### 2. Directory of document files

When `--dataset-path` points to a directory, stfo-colbert will scan for files and create a compressed cache file (`.stfo_colbert_cache.txt.xz`) in that directory. On later runs, this cache is reused instead of re-parsing all files, significantly speeding up initialization.

**Supported file types:**
- `.txt`, `.md`
- `.pdf`

**Cache behavior:**
- The cache file is automatically created after the first directory scan
- To force a re-scan, delete the `.stfo_colbert_cache.txt.xz` file from the dataset directory

## Index Format

stfo-colbert uses PyLate's PLAID index under the hood:
- Loads the model (default: `mixedbread-ai/mxbai-edge-colbert-v0-17m`)
- Encodes documents and builds an index in-memory
- Serves top-k retrieval via a simple HTTP API

The index directory contains:
- **PLAID index files**: The core PyLate index structure
- **`collection.json.xz`**: A compressed JSON mapping of document IDs to their text content

When you build an index from documents, stfo-colbert automatically creates the `collection.json.xz` file to enable text retrieval in search results. If you pass `--index-path` with an existing index, search results will include text snippets only if `collection.json.xz` is present in the index directory.

## HTTP API

### `GET /search`

**Parameters:**
- `query` (string, required): The search string
- `k` (integer, optional): Top-k results (default: `10`, max: `100`)

**Response:**
```json
{
  "query": "...",
  "topk": [
    {
      "pid": "<document_id>",
      "score": 0.95,
      "text": "...",
      "prob": 0.87
    }
  ]
}
```

> **Note:** The `text` field is included if the collection mapping is available (e.g., from a delimited TXT or `collection.json.xz`).

## Design Notes

- **Functional approach**: Modules expose pure functions; the CLI composes them
- **Minimal dependencies**: FastAPI for the web layer, Uvicorn ASGI server, PyLate for model+index, PyMuPDF for PDF parsing
- **Persistent caching**: When processing directories, a compressed cache file (`.stfo_colbert_cache.txt.xz`) is saved in the dataset directory for faster subsequent runs

## Development

**Install in editable mode:**
```bash
pip install -e .
```

**Run tests:**
```bash
pip install pytest
pytest
```

## Examples

### Using the included example data

**Index Wikipedia summaries and query for specific topics:**
```bash
# Start the server with Wikipedia summaries
stfo-colbert --dataset-path example_data/wikipedia_summaries.txt

# Query for movies
curl "http://127.0.0.1:8889/search?query=Disney%20animated%20movies&k=3"

# Query for sports
curl "http://127.0.0.1:8889/search?query=Olympic%20track%20and%20field%20events&k=5"
```

**Index arXiv PDFs and search research papers:**
```bash
# Start the server with PDF directory
stfo-colbert --dataset-path example_data/arxiv_sample

# Search for AI/ML topics
curl "http://127.0.0.1:8889/search?query=machine%20learning%20transformers&k=5"

# Search for specific research areas
curl "http://127.0.0.1:8889/search?query=neural%20network%20architecture&k=3"
```

### General usage examples

**Index directory of Markdown notes and serve on port 7777:**
```bash
stfo-colbert --dataset-path ~/notes --port 7777
```

**Serve existing index folder:**
```bash
stfo-colbert --index-path ./experiments/wiki_index --port 8889
```
