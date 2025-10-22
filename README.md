# stfo-colbert

> Straightforward ColBERT indexing and serving

## Design Goals

- **Straightforward**: Single-command usage via CLI (stfo is for "straightforward")
- **Minimal**: Readable, functional code with minimal default dependencies
- **Simple**: One HTTP endpoint only: `GET /search`
- **Dataset-agnostic**: Accepts a single delimited `.txt` file or a directory of common document files
- **Fast**: Indexing and serving in one process keeps the index in memory for fast queries

## When to Use

Use **stfo-colbert** when you:
- Have a small-to-medium collection and want a simple way to build a sparse ColBERT-style index (via PyLate) and query it over HTTP
- Prefer a one-shot CLI to index and serve, without additional orchestration

## Installation

### From PyPI

```bash
pip install stfo-colbert
```

Or with optional document parser support:

```bash
pip install "stfo-colbert[docs]"
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
curl "http://127.0.0.1:8889/search?query=hello&k=5"
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

When `--dataset-path` points to a directory, stfo-colbert will scan for files and build a temporary delimited TXT for indexing, then delete it afterward.

**Supported file types:**
- **Always supported**: `.txt`, `.md`
- **Optional support** (requires extras): `.pdf`, `.docx`, `.odt`

To enable optional formats:
```bash
pip install "stfo-colbert[docs]"
```

## Index Format

stfo-colbert uses PyLate's PLAID index under the hood:
- Loads the model (default: `mixedbread-ai/mxbai-edge-colbert-v0-17m`)
- Encodes documents and builds an index in-memory
- Serves top-k retrieval via a simple HTTP API

If you pass `--index-path`, it must be a folder containing a previously built PLAID index (index file + metadata). If a matching `collection.tsv` is present next to it, text snippets will be returned.

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

> **Note:** The `text` field is included if the collection mapping is available (e.g., from a delimited TXT or `collection.tsv`).

## Design Notes

- **Functional approach**: Modules expose pure functions; the CLI composes them
- **Minimal dependencies**: FastAPI for the web layer, Uvicorn ASGI server, PyLate for model+index. Optional document parsers are extras
- **Temporary files**: Any temporary delimited TXT created from a directory is deleted automatically after the index is constructed

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

**With optional document parsers:**
```bash
pip install -e ".[docs]"
```

## Examples

**Index directory of Markdown notes and serve on port 7777:**
```bash
stfo-colbert --dataset-path ~/notes --port 7777
```

**Serve existing index folder:**
```bash
stfo-colbert --index-path ./experiments/wiki_index --port 8889
```
