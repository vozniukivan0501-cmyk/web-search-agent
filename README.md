---
title: Web Search Agent
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Web Search Agent

A multi-step reasoning web search agent that plans search queries, pulls live web results via DuckDuckGo, filters them with a local Cross-Encoder to save LLM context tokens, scrapes relevant pages, and synthesizes answers with citations using Google Gemini.

Includes a FastAPI backend with Server-Sent Events (SSE) for streaming graph status updates to a responsive web UI.

---

## Architecture

The agent runs as an iterative state machine compiled with **LangGraph**:

```
[Start]
   │
   ▼
[Plan] ───────► Generates 1-3 targeted sub-queries from user question
   │
   ▼
[Search] ─────► Fetches organic search results from DuckDuckGo
   │
   ▼
[Rerank] ─────► Local Cross-Encoder scores query-snippet pairs; drops noise
   │
   ▼
[Fetch] ──────► Scrapes readable body text from top-scoring URLs
   │
   ▼
[Synthesize] ─► Gemini drafts a referenced answer from scraped content
   │
   ▼
[Evaluate] ───► Verifies if answer satisfies query
   │
   ├── Incomplete & steps < max_reasoning_steps ──► Loops back to [Plan]
   └── Complete or max steps reached ─────────────► [End]
```

### Why Cross-Encoder Reranking?
Feeding raw search results directly into an LLM wastes input tokens on irrelevant ads, navigation text, and low-quality snippets. A local transformer model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each snippet against the query before scraping, keeping LLM prompts concise and focused.

---

## Tech Stack

- **Orchestration:** LangGraph (state graph & conditional loops)
- **LLM Integration:** LlamaIndex (`llama-index-llms-gemini`)
- **Reranker:** Sentence-Transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Web Search & Scraping:** DuckDuckGo (HTML / Lite API), HTTPX, BeautifulSoup4
- **Backend:** FastAPI, Uvicorn, SSE-Starlette
- **Frontend:** Vanilla HTML, CSS, JavaScript (marked.js + highlight.js)
- **Containerization:** Docker

---

## Quickstart

### Option 1: Local Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd web-search-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Add your Google Gemini API key to `.env`:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```
   Open [http://localhost:8000](http://localhost:8000) in your browser.

---

### Option 2: Docker

1. **Build the image:**
   ```bash
   docker build -t web-search-agent .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name search-agent \
     -p 8000:8000 \
     -e GOOGLE_API_KEY="your_gemini_api_key_here" \
     web-search-agent
   ```
   The web interface will be accessible at [http://localhost:8000](http://localhost:8000).

---

## Configuration

Configuration values can be adjusted via environment variables or inside `.env`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | *(None)* | Server default Gemini API key (optional if provided in UI) |
| `MODEL_NAME` | `models/gemini-2.0-flash` | Gemini model identifier used for planning and synthesis |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face model used for local snippet reranking |
| `MAX_SEARCH_RESULTS` | `10` | Max raw search results to retrieve per sub-query |
| `TOP_K_RERANKED` | `3` | Number of top reranked pages to fetch and read |
| `MAX_REASONING_STEPS` | `5` | Loop limit for multi-step refinement |
| `HOST` | `0.0.0.0` | Bind host address |
| `PORT` | `8000` | Port for the FastAPI server |

> **Note on API Keys:** If `GOOGLE_API_KEY` is configured on the server, visitors do not need to provide their own key in the browser. If a user enters a key in the header input, their key overrides the server default for that session.

---

## API Endpoints

- `POST /api/search` — Runs the full graph pipeline and returns JSON containing `answer`, `sources`, and `reasoning_steps`.
- `POST /api/search/stream` — Runs the pipeline and streams status changes and output chunks via SSE (`text/event-stream`).
- `GET /api/config` — Returns public server configuration (e.g., whether a default API key is set).
- `GET /api/health` — Basic health check endpoint.

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py           # Pydantic settings loading from .env
│   ├── agent.py            # High-level runners (invoke & async generator for SSE)
│   ├── graph.py            # LangGraph nodes, state definitions, and graph assembly
│   ├── cross_encoder.py    # Local Sentence-Transformers model caching and scoring
│   ├── search_tools.py     # DuckDuckGo query execution and HTML page text extraction
│   └── server.py           # FastAPI app, SSE endpoint, and static file mounting
├── static/
│   ├── index.html          # Web UI layout
│   ├── styles.css          # Dark UI styling
│   └── app.js              # SSE consumer, markdown renderer, and UI logic
├── Dockerfile
├── requirements.txt
├── .env.example
├── main.py                 # Application entry point
└── README.md
```

---

## Hardware Requirements

- **RAM:** Minimum 1.5–2 GB required due to PyTorch and the Cross-Encoder model. Free tiers on shared platforms limited to 512 MB may run into OOM errors.
- **CPU:** Works on standard x86_64 / ARM64 CPUs. GPU is not required.
