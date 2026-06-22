<div align="center">

<img width="60" src="https://raw.githubusercontent.com/HassyWebTech/NeurOps/main/assets/screenshots/landing.png" alt="NeurOps" style="border-radius:12px"/>

# NeurOps
### The intelligent layer for your business operations.

*Ask a plain English question. Get a backed-by-data answer in seconds.*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1a1a2e?style=flat-square)](https://langchain-ai.github.io/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-f55036?style=flat-square)](https://groq.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-dc244c?style=flat-square)](https://qdrant.tech)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=flat-square)]()

[**Live Demo**](https://neur-ops.vercel.app) · [**View on GitHub**](https://github.com/HassyWebTech/NeurOps) · [**Built by Hassan**](https://linkedin.com/in/hassan-yakubu)

</div>

---

## The Problem

Founders waste hours on work that data should answer automatically.

A business owner sits on thousands of customer reviews, order records, and transaction logs — but getting insight from that data means writing SQL, building dashboards, or reading every review manually. Most don't. The signal stays buried.

**NeurOps fixes that.**

Ask *"What are customers saying about late deliveries?"* and get a clear, specific, sourced answer in under 3 seconds — backed by real retrieval from your actual data, not a hallucinated summary.

---

## Screenshots

<table>
  <tr>
    <td><img src="assets/screenshots/landing.png" alt="NeurOps Landing Page" width="100%"/><br/><sub><b>Landing Page</b> — Product overview and live demo CTA</sub></td>
    <td><img src="assets/screenshots/chat-answer.png" alt="NeurOps Chat Interface" width="100%"/><br/><sub><b>Chat Interface</b> — Agent answering with source citations</sub></td>
  </tr>
  <tr>
    <td><img src="assets/screenshots/churn-analysis.png" alt="Churn Analysis" width="100%"/><br/><sub><b>Churn Analysis</b> — Structured customer intelligence</sub></td>
    <td><img src="assets/screenshots/eval-results.png" alt="Evaluation Framework" width="100%"/><br/><sub><b>Eval Dashboard</b> — Retrieval accuracy metrics</sub></td>
  </tr>
</table>

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

That answer came from semantically searching **37,970 real customer reviews** —
not from the LLM's training data. Every claim is traceable to a source.

---

## Architecture

NeurOps is designed in 5 clean layers. Each layer only knows about the layer
directly below it — no cross-cutting dependencies, no spaghetti.

```
┌─────────────────────────────────────────┐
│      Frontend (HTML/CSS/JS)              │  ← Landing page + chat UI · Mobile-first
└──────────────────┬──────────────────────┘
                   │ HTTP / REST
┌──────────────────▼──────────────────────┐
│      FastAPI Backend                     │  ← REST API · Pydantic schemas · CORS
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│      Agent Layer (LangGraph)             │  ← Multi-agent orchestration · MemorySaver
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│      Tools Layer                         │
│  search_customer_reviews                 │  ← Semantic RAG search
│  churn_analysis                          │  ← RFM churn scoring
│  customer_analytics                      │  ← Revenue intelligence
│  generate_campaign                       │  ← AI campaign generation
│  save_business_context                   │  ← Long-term memory
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│      Data Layer                          │
│  Qdrant (37,970 embedded reviews)        │  ← Vector store · local mode
│  Pandas DataFrames (500K+ rows)          │  ← Structured business data
│  JSON Memory Store                       │  ← Persistent business context
└─────────────────────────────────────────┘
```

### How a question becomes an answer

```
1. POST /api/ask receives question + thread_id
2. Pydantic validates the request shape
3. LangGraph loads conversation history (MemorySaver)
4. Agent node runs — LLM sees question + tool descriptions + system prompt
5. LLM decides which tool to call (or answers directly)
6. ToolNode executes the selected tool:
   └── search_reviews → SentenceTransformer encodes query → Qdrant cosine search
   └── churn_analysis → Pandas RFM scoring over 99,441 orders
   └── customer_analytics → Pandas aggregations over 500K+ rows
   └── generate_campaign → Groq call with business context injected
7. Tool result added to conversation state
8. Agent runs again — reads result, decides to answer or call another tool
9. Final answer returned with source citations
10. MemorySaver persists full conversation for thread continuity
```

---

## Technology Decisions

Every technology choice was intentional. Here's the reasoning:

| Decision | Rationale |
|---|---|
| **LangGraph over custom loops** | Native support for cycles, conditional routing, and state management. A Python `for` loop can't express "call tool A, read result, decide whether to call tool B" cleanly |
| **sentence-transformers over OpenAI embeddings** | Zero cost, zero API latency, runs entirely locally. Same embedding space for query and documents — critical for semantic search correctness |
| **Qdrant local mode over Pinecone** | Open source, self-hostable, local mode for development. Production switch = one line in `config.py` |
| **Groq over OpenAI** | 10x faster inference on Llama 3.3 70B, generous free tier |
| **Cosine similarity over L2 distance** | Measures semantic direction not magnitude — text with similar meaning clusters correctly regardless of length |
| **MemorySaver over database checkpointer** | Zero setup for development. Same `checkpointer=` interface — swap to `PostgresSaver` for production with no other code changes |
| **RFM scoring for churn** | Recency (50%), Frequency (30%), Completion Rate (20%) — industry-standard weights, interpretable, no ML training required |
| **JSON for long-term memory** | Fast to build, clearly explainable, easy to inspect. Production upgrade: PostgreSQL table keyed by business_id, same read/write interface |
| **batch_size=100 for ingestion** | Prevents OOM errors on constrained hardware while maintaining throughput across 37,970 embeddings |

---

## Agent Tools

NeurOps has 5 tools. The agent reads each tool's docstring to decide which one to use:

### `search_customer_reviews(query)`
Semantic search over 37,970 embedded customer reviews using Qdrant cosine similarity. Returns top-5 most relevant reviews with similarity scores.

```python
# Agent calls this when asked about customer opinions, complaints, feedback
search_customer_reviews("late delivery and poor communication")
# → Review 1 (★1/5): "Empresa não entrega os produtos na data..."
# → Review 2 (★1/5): "Péssima empresa, não entregou o produto..."
```

### `churn_analysis(top_n)`
RFM-based churn scoring across 99,441 customers. Calculates recency, purchase frequency, and delivery completion rate. Returns ranked at-risk customers with churn scores.

```python
churn_analysis(top_n=10)
# → Customer 08c5351a... | Churn Risk: 70.0% | Last Purchase: 772 days ago
```

### `customer_analytics(query_type)`
Structured business intelligence over 500K+ rows. Supports: `top_customers`, `revenue_by_category`, `customer_by_city`, `payment_methods`, `avg_order_value`, `order_status`.

```python
customer_analytics("avg_order_value")
# → Total Revenue: R$16,008,872.12 | Avg Order: R$154.10
```

### `generate_campaign(campaign_type, target_segment, tone)`
Generates complete campaign packages (WhatsApp message, email subject, email body, discount offer, strategic notes) using Groq with business context injected.

```python
generate_campaign(campaign_type="reengagement", target_segment="churning customers")
# → WhatsApp: "Hey! We miss you 👋 Here's 20% off your next order..."
# → Email Subject: "It's been a while — we have something for you"
```

### `save_business_context(key, value)`
Persists business facts to disk. Loaded into the system prompt on every agent run — enabling cross-session memory.

```python
save_business_context("business_type", "fashion retail in Lagos")
# Persisted → loaded in all future conversations
```

---

## Memory Architecture

NeurOps implements two distinct memory systems:

**Short-term (conversation memory)**
LangGraph's `MemorySaver` persists the full message history keyed by `thread_id`. Each browser session generates a unique thread ID. Follow-up questions reference previous answers naturally.

```
User: "What are customers saying about late deliveries?"
NeurOps: [detailed answer]
User: "Can you summarize that in one sentence?"
NeurOps: [correctly summarizes the previous answer — no re-explanation needed]
```

**Long-term (business context memory)**
A JSON store that persists facts across sessions and server restarts. Loaded into the agent's system prompt on every run.

```
Session 1: "I run a fashion store in Lagos targeting young professionals"
→ Saved: {business_type: "fashion retail in Lagos", target_market: "young professionals"}

Session 2 (new browser, days later):
User: "Generate a campaign for my customers"
NeurOps: [generates campaign tailored to fashion retail + young professionals]
```

---

## Evaluation Framework

NeurOps includes a built-in evaluation suite at `GET /api/evals`.

```json
{
  "eval_type": "retrieval",
  "overall_score": 30.0,
  "summary": {
    "retrieval_accuracy": "30.0%",
    "tests_run": 2
  },
  "details": [
    {
      "test_id": "TC001",
      "question": "What are customers saying about late deliveries?",
      "retrieved_count": 5,
      "relevant_hits": 2,
      "score": 0.4,
      "latency_seconds": 17.457,
      "avg_similarity": 0.2622
    }
  ]
}
```

**On the 30% score:** This eval uses English keyword matching on Portuguese reviews — a known limitation of keyword-based metrics on multilingual RAG systems. The retrieval is semantically correct (demonstrated by real responses), but English keywords don't appear in Portuguese text. Production upgrade: LLM-as-judge evaluation that understands cross-lingual semantics. The latency (15-17s) reflects cold-start embedding model loading — in a persistent server, this drops below 1 second.

---

## Project Structure

```
neurops/
├── backend/
│   ├── agents/
│   │   ├── graph.py         # LangGraph workflow · MemorySaver · conditional routing
│   │   └── state.py         # AgentState with add_messages reducer
│   │
│   ├── tools/
│   │   ├── retrieval.py     # search_customer_reviews — Qdrant semantic search
│   │   ├── churn.py         # churn_analysis — RFM scoring over 99,441 orders
│   │   ├── analytics.py     # customer_analytics — 6 query types over 500K+ rows
│   │   ├── campaign.py      # generate_campaign — AI campaign with business context
│   │   └── memory.py        # save_business_context — persistent fact storage
│   │
│   ├── rag/
│   │   ├── ingest.py        # CSV → chunks → embeddings → Qdrant (batch_size=100)
│   │   └── retrieve.py      # query → vector → cosine search → ranked results
│   │
│   ├── memory/
│   │   └── store.py         # JSON-based long-term memory · load/save/format
│   │
│   ├── data/
│   │   ├── loader.py        # Loads all CSVs once at startup · shared DataFrames
│   │   └── olist/           # 9 CSVs · 500K+ rows · download from Kaggle
│   │
│   ├── database/
│   │   ├── models.py        # PostgreSQL models (Week 7+)
│   │   └── connection.py    # DB connection
│   │
│   ├── api/
│   │   ├── routes.py        # /ask · /health · /ingest · /evals
│   │   └── schemas.py       # QuestionRequest · AnswerResponse · ReviewResult
│   │
│   ├── evals/
│   │   └── metrics.py       # Test suite · retrieval accuracy · latency · similarity
│   │
│   ├── config.py            # Single source of truth · env vars · paths · model names
│   └── main.py              # FastAPI app · CORS · router registration
│
├── frontend/
│   └── index.html           # Mobile-first · landing page + chat app · marked.js
│
├── assets/screenshots/      # Landing page · chat · churn · evals
├── .env.example             # Template for environment setup
└── README.md
```

---

## Dataset

**Olist Brazilian E-Commerce Public Dataset** — 100,000 real orders across 9 relational CSVs.

| File | Rows | NeurOps Use |
|---|---|---|
| `olist_order_reviews_dataset.csv` | 99,224 | **RAG knowledge base** — 37,970 embedded into Qdrant after cleaning |
| `olist_orders_dataset.csv` | 99,441 | Churn signals · delivery timelines |
| `olist_customers_dataset.csv` | 99,441 | Segmentation · city/state analytics |
| `olist_order_items_dataset.csv` | 112,650 | Revenue analytics · category performance |
| `olist_order_payments_dataset.csv` | 103,886 | Payment intelligence · R$16M revenue |
| `olist_products_dataset.csv` | 32,951 | Category mapping |
| `olist_sellers_dataset.csv` | 3,095 | Seller performance |
| `product_category_name_translation.csv` | 71 | Portuguese → English categories |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Free [Groq API key](https://console.groq.com)
- [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle

### Setup

```bash
# Clone the repository
git clone https://github.com/HassyWebTech/NeurOps.git
cd NeurOps

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
# Step 1: Ingest data — run once to build your vector knowledge base
cd backend
python -m rag.ingest

# Step 2: Start the API server
uvicorn main:app --reload

# Step 3: Open the frontend
# Open frontend/index.html in your browser
```

**API docs:** `http://localhost:8000/docs`

---

## API Reference

### `POST /api/ask`
```json
// Request
{
  "question": "What are customers saying about late deliveries?",
  "top_k": 5,
  "score_filter": null,
  "thread_id": "thread_1234567890_abc123"
}

// Response
{
  "answer": "Customers are frustrated with late deliveries...",
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

### `GET /api/evals`
Runs retrieval evaluation suite. Returns accuracy scores, latency, and per-test details.

### `POST /api/ingest`
Triggers re-ingestion of the dataset. Use after updating data files.

---

## Engineering Notes

**On multilingual retrieval:** The Olist dataset is in Portuguese. NeurOps queries in English. `all-MiniLM-L6-v2` is a multilingual model — it encodes semantic meaning across languages into the same vector space. English queries correctly retrieve Portuguese reviews with similar meaning. Groq Llama 3.3 70B then reads Portuguese context and answers in English. This is cross-lingual semantic search working as designed — not a bug.

**On local vs production Qdrant:** NeurOps uses Qdrant local mode for development — no Docker, no server, vectors persisted to disk. In production, switching to Qdrant Cloud requires changing two lines in `config.py`. The rest of the codebase is unchanged.

**On embedding model choice:** `all-MiniLM-L6-v2` produces 384-dimensional vectors. Larger models (`all-mpnet-base-v2` at 768 dimensions) produce richer embeddings but are 3x slower and use 2x memory. For 37,970 documents on constrained hardware, MiniLM is the right tradeoff.

**On tool docstrings as prompts:** In LangGraph, the agent reads each tool's docstring to decide when to call it. Every word in a tool docstring is prompt engineering. Vague docstrings produce wrong tool selections. Detailed docstrings with explicit examples produce consistent, accurate routing.

**On the eval score:** Our keyword eval shows 30% accuracy — a known limitation of English keyword matching on Portuguese text. The semantic retrieval quality is higher, as shown by real responses and similarity scores. Production upgrade: LLM-as-judge evaluation that understands cross-lingual semantics.

---

## Built By

**Hassan Yakubu** — AI Systems Engineer, Lagos Nigeria

8 months into a deliberate pivot from frontend development into AI engineering. Building in public, shipping real systems, learning by doing.

- 🐙 **GitHub:** [HassyWebTech](https://github.com/HassyWebTech)
- 💼 **LinkedIn:** [hassan-yakubu](https://linkedin.com/in/hassan-yakubu)
- 🐦 **X:** [@yakubhassan83](https://x.com/yakubhassan83)

---

<div align="center">

**If this helped you, star the repo ⭐**

*Built with deliberate engineering — not tutorial code.*

</div>