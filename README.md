# Enterprise Knowledge Assistant (RAG)

A question-answering assistant over a set of internal documents. Documents are
chunked, embedded, and indexed in **Azure AI Search**; questions are answered by
**Azure OpenAI** using only the retrieved chunks, with inline citations. The
whole pipeline is exposed through a **FastAPI** endpoint.

**Stack:** Python 3.12 · FastAPI · Azure OpenAI · Azure AI Search

---

## What it does

```
question ──► embed ──► hybrid search (BM25 + vector) ──► semantic rerank
                                                              │
                    grounded prompt ◄── top-k chunks ◄─────────┘
                            │
                            └──► Azure OpenAI chat ──► answer + citations
```

- **Chunking** — token-aware, paragraph-respecting, with configurable overlap
  (`app/chunking.py`).
- **Retrieval** — hybrid search combining keyword and vector scoring, with the
  Azure semantic reranker on top when the search tier supports it
  (`app/retrieval.py`).
- **Grounding** — the model is given numbered sources and instructed to answer
  only from them, or to say it couldn't find the answer (`app/prompts.py`).
- **Citations** — every response carries the chunks it was built from, with
  source file, chunk index, and retrieval score, so an answer can be checked.

## Why grounding, not just prompting

Asking the chat model directly produced confident answers to questions the
document set never covered — the classic failure mode this project exists to fix.
Three changes cut off-topic answers sharply:

1. **Retrieve first, answer second.** If retrieval returns nothing, the pipeline
   returns the refusal string without calling the model at all (`app/rag.py`).
   A model handed no context will answer from pretraining every time.
2. **Numbered sources.** Un-numbered context produced answers citing nothing, or
   inventing plausible-looking document names.
3. **An explicit refusal instruction** with a fixed string, so "I don't know" is
   detectable programmatically rather than being phrased differently each time.

`data/eval_questions.md` holds the hand-written question set used to check this
after every change to chunk size, `top_k`, or the prompt. Questions 7–9 are
deliberately unanswerable from the corpus and must come back refused.

## Tuning notes

Two knobs moved answer quality most, and both are settings rather than constants:

- **Chunk size (default 500 tokens).** Smaller chunks retrieved precisely but
  cut definitions in half; larger chunks buried the relevant sentence in noise
  and pushed the question out of the model's focus.
- **Overlap (default 100 tokens).** Facts that straddle a chunk boundary — a
  threshold in one paragraph, its exception in the next — were the most common
  retrieval miss. The overlap carries the tail of each chunk into the next one.

Chunk sizing is pure and offline, so it can be tested without Azure: see
`tests/test_chunking.py`.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in your Azure endpoints and keys
```

You need, in Azure:

- An **Azure OpenAI** resource with a chat deployment (e.g. `gpt-4o-mini`) and an
  embedding deployment (e.g. `text-embedding-3-small`, 1536 dimensions).
- An **Azure AI Search** service. The Basic tier and above support the semantic
  ranker; on Free, set `USE_SEMANTIC_RANKER=false` (the code also falls back on
  its own if the ranker call fails).

## Index your documents

Drop `.md`, `.txt`, or `.pdf` files into `data/docs/` (three sample policy
documents are included), then:

```bash
python -m scripts.ingest --docs data/docs
```

This creates the search index if needed, chunks each document, embeds the chunks
in batches, and uploads them.

## Run the API

```bash
uvicorn app.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

### `POST /ask`

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How long do I have to submit an expense claim?"}'
```

```json
{
  "answer": "Expense claims must be submitted within 30 calendar days of the date the expense was incurred. Claims submitted after 60 days are not reimbursed without written approval from a department head [1].",
  "citations": [
    {
      "n": 1,
      "source": "expense-policy.md",
      "title": "expense policy",
      "chunk_index": 1,
      "score": 2.6231,
      "excerpt": "Expense claims must be submitted within 30 calendar days..."
    }
  ],
  "grounded": true,
  "latency_ms": 1142
}
```

Optional fields: `top_k` (override retrieval depth), `source` (restrict to one
document).

### `GET /search`

Retrieval only — shows which chunks a question pulls, without generating an
answer. This is the endpoint to reach for when an answer looks wrong: it
separates a retrieval problem from a generation problem.

```bash
curl -s 'localhost:8000/search?q=alcohol%20reimbursement&top_k=3'
```

### `GET /health`

Reports the configured index and deployments, and whether credentials are
present.

## Tests

```bash
pytest
```

The suite runs without Azure credentials — chunking and prompt construction are
pure functions, and the API and pipeline tests stub the Azure clients.

## Docker

```bash
docker build -t knowledge-assistant .
docker run -p 8000:8000 --env-file .env knowledge-assistant
```

## Layout

```
app/
  main.py          FastAPI routes (/ask, /search, /health)
  rag.py           retrieve -> ground -> generate
  retrieval.py     hybrid + semantic search against Azure AI Search
  chunking.py      document loading and token-aware chunking
  ingest.py        chunk -> embed -> upload
  index_setup.py   search index schema (vector + semantic config)
  prompts.py       system prompt and context formatting
  azure_clients.py cached Azure OpenAI / Search clients
  config.py        settings
scripts/ingest.py  ingestion CLI
data/docs/         sample corpus
data/eval_questions.md
tests/
```
