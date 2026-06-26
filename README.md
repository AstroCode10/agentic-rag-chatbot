# Agentic RAG Chatbot

A retrieval-augmented generation chatbot built from first principles. Given a document corpus, it chunks, embeds, and indexes the content locally, then uses an LLM as an agent to decide when to search the knowledge base and when to answer directly.

Built as a portfolio project for the AI Engineering track (Week 3 milestone).

---

## What it does

A user asks a question in natural language. An agent backed by an LLM decides whether the question requires searching the knowledge base or can be answered directly. If it searches, it runs a two-stage retrieval pipeline — fast embedding search followed by cross-encoder re-ranking — assembles the top chunks into context, and generates a grounded answer with citations. The full pipeline is evaluated with RAGAS across three metrics.

---

## Pipeline overview

```
Document
   │
   ▼
Step 1 — Chunking          Split into overlapping character or sentence chunks
   │
   ▼
Step 2 — Embed + Store     Encode chunks with SentenceTransformer → store in ChromaDB
   │
   ▼
Step 3 — Retrieve + Rerank  Embed query → cosine search (top 20) → cross-encoder rerank (top 5)
   │
   ▼
Step 4 — Agent             LLM decides: call search_docs tool or answer directly
   │
   ▼
Step 5 — Evaluate          RAGAS scores: faithfulness, answer relevance, context recall
```

---

## Stack

| Component | Library | Purpose |
|---|---|---|
| Embedding | `sentence-transformers` (`all-MiniLM-L6-v2`) | Text → 384-dim vectors |
| Vector store | `chromadb` | Persistent local vector database |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precise relevance scoring |
| Generation | OpenAI API (`gpt-oss-120b`) | Agent + answer generation |
| Evaluation | `ragas` | Faithfulness, relevance, recall metrics |

Everything except the Anthropic API runs locally. No external vector database accounts required.

---

## Project structure

```
.
├── step1_chunking.py          # Document splitting — character and sentence strategies
├── step2_embed_and_store.py   # Embedding + ChromaDB ingestion
├── step3_retrieve_and_rerank.py  # Two-stage retrieval pipeline
├── step4_agent.py             # LLM agent with search_docs tool
├── step5_evaluate.py          # RAGAS evaluation suite
├── chatbot.py                 # Entry point — conversational loop
├── chroma_db/                 # Auto-created by ChromaDB on first ingest
└── README.md
```

---

## Setup

**Install dependencies**

```bash
pip install chromadb sentence-transformers ragas
```

**Set your OpenRouter API key**

```bash
export OPENROUTER_API_KEY=your_key_here
```

**Ingest a document**

```python
from step1_chunking import chunk_text
from step2_embed_and_store import build_knowledge_base

with open("your_document.txt") as f:
    text = f.read()

chunks = chunk_text(text, chunk_size=500, overlap=100)
build_knowledge_base(chunks, collection_name="my_docs")
```

**Run the chatbot**

```bash
python chatbot.py
```

---

## Key design decisions

**Why overlapping chunks?**
A critical sentence sitting at a chunk boundary would be split across two chunks, making it invisible to either. A 100-character overlap ensures ideas that straddle boundaries appear in at least one complete chunk.

**Why two-stage retrieval?**
Embedding search (bi-encoder) is fast but approximate — it finds chunks on the same topic, not necessarily chunks that answer the specific question. The cross-encoder re-ranker reads the query and each candidate together, catching cases where different wording masks a perfect answer. Running the cross-encoder on all chunks would be too slow; running it on the top 20 from embedding search costs milliseconds.

**Why an agent rather than always retrieving?**
Some questions (greetings, general knowledge, clarifications) don't benefit from retrieval — injecting irrelevant chunks degrades the answer. Giving Claude a `search_docs` tool and letting it decide produces cleaner responses and makes the system's reasoning transparent in the tool call logs.

---

## Evaluation

The system is evaluated with RAGAS on a hand-curated test set of 15 question/answer pairs.

| Metric | What it measures |
|---|---|
| Faithfulness | Are all claims in the answer grounded in retrieved chunks? |
| Answer relevance | Does the answer address what was actually asked? |
| Context recall | Did retrieval surface the chunks needed to answer correctly? |

**Known failure modes**

*Mid-sentence chunk boundaries* — character-based chunking can split a definition or argument across two chunks. Neither chunk alone scores well on context recall for questions about that idea. Mitigation: sentence-based chunking or larger overlap.

*Embedding–relevance mismatch* — the embedding model retrieves chunks that are topically adjacent but don't answer the specific question. The cross-encoder catches most of these, but on highly technical or narrow queries the top-20 candidate pool can contain zero truly relevant chunks, making re-ranking irrelevant. Mitigation: larger initial retrieval pool or hybrid BM25 + dense search.

---

## Extending this project

- **Hybrid search** — combine dense (embedding) and sparse (BM25) retrieval before re-ranking for better recall on keyword-heavy queries
- **Multi-document support** — tag chunks with source metadata and filter by document at query time
- **Streaming responses** — use the Anthropic streaming API for a more responsive chat experience
- **Web interface** — wrap `chatbot.py` in a Flask or Streamlit app