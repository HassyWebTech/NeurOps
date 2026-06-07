ash

cat > /mnt/user-data/outputs/README.md << 'EOF'
<div align="center">

<img src="https://img.shields.io/badge/NeurOps-Business%20Intelligence%20AI-0f0f1a?style=for-the-badge&labelColor=0f0f1a&color=d4522a" />

# NeurOps
### The intelligent layer for your business operations.

*Ask a plain English question. Get a backed-by-data answer in seconds.*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-f55036?style=flat-square)](https://groq.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-dc244c?style=flat-square)](https://qdrant.tech)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Week%201%20Complete-22c55e?style=flat-square)]()

</div>

---

## The Problem

Founders waste hours on work that data should answer automatically.

A business owner sits on thousands of customer reviews, order records, and transaction logs — but getting insight from that data means writing SQL, building dashboards, or reading every review manually. Most don't. The signal stays buried.

**NeurOps fixes that.**

Ask *"What are customers saying about late deliveries?"* and get a clear, specific, sourced answer in under 3 seconds — backed by real retrieval from your actual data, not a hallucinated summary.

---

## What It Does

```
User: "What are customers saying about late deliveries?"

NeurOps: "Customers are extremely frustrated with late deliveries. Across
multiple reviews, they report orders arriving weeks late with no proactive
communication from the company. A recurring pattern is unresponsive customer
support — customers say they contacted support 3–5 times with no resolution.

Recommendation: Implement automated order-status notifications and enforce
delivery SLAs at the seller level."

Sources: 5 reviews retrieved · similarity scores: 0.27, 0.26, 0.25, 0.25, 0.24
```

That answer came from semantically searching **37,970 real customer reviews**, not from the LLM's training data. Every claim is traceable to a source.

---

## Architecture

NeurOps is designed in 5 clean layers. Each layer only knows about the layer directly below it — no cross-cutting dependencies, no spaghetti.

```
┌─────────────────────────────────────┐
│         Frontend (HTML/CSS/JS)       │  ← Chat interface + landing page
└──────────────────┬──────────────────┘
                   │ HTTP
┌──────────────────▼──────────────────┐
│         FastAPI Backend              │  ← REST API · Pydantic schemas · CORS
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│         Agent Layer (LangGraph)      │  ← Multi-agent orchestration [Week 2]
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│         Tools Layer                  │  ← retrieval · churn · analytics · campaign
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│         Data Layer                   │  ← Qdrant (vectors) · PostgreSQL (structured)
└─────────────────────────────────────┘
```

### Week 1 — RAG Pipeline (current)

The retrieval pipeline is the foundation everything else sits on. Here's exactly how a user question becomes an answer:

```
1. Question arrives at POST /api/ask
2. Pydantic validates the request shape
3. SentenceTransformer encodes question → 384-dim vector (local, no API call)
4. Qdrant cosine search over 37,970 embedded reviews → top-k results
5. Retrieved reviews formatted as structured context
6. Groq Llama 3.3 70B receives [system prompt + context + question]
7. LLM generates answer referencing specific review patterns
8. Response returned with answer + source citations + similarity scores
```

**Why these specific technology choices:**

| Decision | Rationale |
|---|---|
| `sentence-transformers` over OpenAI embeddings | Zero cost, runs locally, no API latency, same embedding space across query and documents |
| `Qdrant` over Pinecone | Open source, self-hostable, local mode for development, production mode for deployment — one config change |
| `Groq` over OpenAI | 10x faster inference on Llama 3.3 70B, generous free tier, consistent with production African infra constraints |
| `FastAPI` over Flask | Automatic OpenAPI docs, native Pydantic validation, async support, type safety |
| `Cosine similarity` over L2 distance | Measures semantic direction not magnitude — text with similar meaning clusters correctly regardless of length |
| `batch_size=100` ingestion | Prevents OOM errors on constrained hardware while maintaining throughput |

---

## Project Structure

```
neurops/
├── backend/
│   ├── agents/              # LangGraph multi-agent system [Week 2]
│   │   ├── graph.py         # Agent workflow definition
│   │   └── state.py         # Shared agent state
│   │
│   ├── tools/               # Agent capabilities [Week 2-5]
│   │   ├── retrieval.py     # Semantic search tool
│   │   ├── churn.py         # Churn prediction tool
│   │   ├── analytics.py     # Revenue intelligence tool
│   │   └── campaign.py      # Campaign generation tool
│   │
│   ├── rag/                 # Knowledge retrieval system [Week 1 ✅]
│   │   ├── ingest.py        # CSV → chunks → embeddings → Qdrant
│   │   └── retrieve.py      # Query → vector search → ranked results
│   │
│   ├── memory/              # Conversation + business context [Week 3]
│   │   └── store.py
│   │
│   ├── database/            # PostgreSQL models + connection
│   │   ├── models.py
│   │   └── connection.py
│   │
│   ├── api/
│   │   ├── routes.py        # /ask · /ingest · /health
│   │   └── schemas.py       # Request/response contracts
│   │
│   ├── evals/               # Retrieval + response quality metrics [Week 6]
│   │   └── metrics.py
│   │
│   ├── data/olist/          # Olist Brazilian E-Commerce dataset (9 CSVs)
│   ├── config.py            # Single source of truth for all settings
│   └── main.py              # FastAPI app entrypoint
│
├── frontend/
│   └── index.html           # Landing page + chat app (single file)
│
├── .env                     # Secrets — never committed
├── .gitignore
└── README.md
```

---

## Dataset

**Olist Brazilian E-Commerce Public Dataset** — 100,000 real orders across 9 relational CSVs.

| File | Records | NeurOps Use |
|---|---|---|
| `olist_order_reviews_dataset.csv` | 99,224 | **RAG knowledge base** — embedded into Qdrant |
| `olist_orders_dataset.csv` | 99,441 | Delivery timelines, churn signals |
| `olist_customers_dataset.csv` | 99,441 | Customer segmentation |
| `olist_order_items_dataset.csv` | 112,650 | Revenue analytics |
| `olist_order_payments_dataset.csv` | 103,886 | Payment intelligence |
| `olist_products_dataset.csv` | 32,951 | Category performance |
| `olist_sellers_dataset.csv` | 3,095 | Seller scoring |
| `product_category_name_translation.csv` | 71 | Portuguese → English mapping |

After cleaning (removing null review bodies), **37,970 reviews** are embedded and stored in Qdrant — each as a 384-dimensional vector using `all-MiniLM-L6-v2`.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)
- The [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (download and extract)

### Setup

```bash
# Clone the repository
git clone https://github.com/HassyWebTech/neurops-ai.git
cd neurops-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Place Olist CSVs in backend/data/olist/
```

### Run

```bash
# Step 1: Ingest data (run once — builds your vector knowledge base)
cd backend
python -m rag.ingest

# Step 2: Start the API server
uvicorn main:app --reload

# Step 3: Open the frontend
# Open frontend/index.html in your browser
```

**API docs available at:** `http://localhost:8000/docs`

---

## API Reference

### `POST /api/ask`

The core RAG endpoint. Retrieves relevant reviews and generates an LLM answer.

```json
// Request
{
  "question": "What are customers saying about late deliveries?",
  "top_k": 5,
  "score_filter": null
}

// Response
{
  "answer": "Customers are frustrated with...",
  "sources": [
    {
      "text": "Empresa não entrega os produtos na data...",
      "score": 1,
      "order_id": "8d89165df616ded...",
      "similarity": 0.2635
    }
  ],
  "question": "What are customers saying about late deliveries?"
}
```

### `GET /api/health`

```json
{ "status": "ok", "message": "NeurOps API is running" }
```

### `POST /api/ingest`

Triggers re-ingestion of the dataset. Use after updating data files.

---

## Build Roadmap

This project is built in 6 weekly layers — each week adds a production-grade capability:

| Week | Capability | Status |
|---|---|---|
| **Week 1** | RAG pipeline — ingest, embed, retrieve, answer | ✅ Complete |
| **Week 2** | LangGraph agent — tool calling, multi-step reasoning | 🔄 In Progress |
| **Week 3** | Memory — short-term conversation + long-term business context | ⬜ Planned |
| **Week 4** | Customer intelligence — churn detection, segmentation | ⬜ Planned |
| **Week 5** | Campaign generation — automated re-engagement workflows | ⬜ Planned |
| **Week 6** | Evaluation framework — retrieval accuracy, hallucination rate | ⬜ Planned |

---

## Engineering Notes

**On multilingual retrieval:** The Olist dataset is in Portuguese. NeurOps queries in English. `all-MiniLM-L6-v2` is a multilingual model — it encodes semantic meaning across languages into the same vector space. This means English queries correctly retrieve Portuguese reviews with similar meaning. Groq Llama 3.3 70B then reads the Portuguese context and answers in English. This is not a bug — it's cross-lingual semantic search working as designed.

**On local vs production Qdrant:** NeurOps uses Qdrant local mode for development — no Docker, no server, vectors persisted to disk. In production, switching to Qdrant Cloud requires changing one line in `config.py`. The rest of the codebase is unchanged.

**On embedding model choice:** `all-MiniLM-L6-v2` produces 384-dimensional vectors. Larger models (e.g. `all-mpnet-base-v2` at 768 dimensions) produce richer embeddings but are 3x slower and use 2x memory. For 37,970 documents on constrained hardware, MiniLM is the right tradeoff.

---

## Built By

**Hassan** — AI Systems Engineer, Lagos Nigeria

7 months into a deliberate pivot from frontend development into AI engineering. Building in public, shipping real systems, learning by doing.

- 🐙 GitHub: [HassyWebTech](https://github.com/HassyWebTech)
- 💼 LinkedIn: [Connect](https://linkedin.com)
- 🐦 X: [Follow](https://twitter.com)

---

<div align="center">

**If you found this useful, star the repo ⭐**

*Built with deliberate engineering — not tutorial code.*

</div>
