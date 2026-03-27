# Voice AI Agent

A modular, production-oriented **voice-enabled Retrieval-Augmented Generation (RAG)** system for document question answering.

The application ingests PDFs, retrieves relevant context using semantic search, generates accurate responses using LLMs, and delivers them as natural-sounding audio.

Supports both **cloud-based (OpenAI)** and upcoming **local (Ollama)** inference modes for flexible, cost-aware deployment.

---

## 🚀 Key Capabilities

* **Document Intelligence**

  * PDF ingestion with chunking and metadata enrichment
  * Semantic search powered by Qdrant vector database

* **RAG Pipeline**

  * FastEmbed for efficient embeddings
  * Context-aware retrieval (top-k similarity search)
  * Two-stage response generation for improved accuracy and TTS quality

* **Voice Output**

  * OpenAI Text-to-Speech (TTS)
  * In-browser playback + downloadable MP3

* **Modular Architecture**

  * Clean `src/`-based structure
  * Separation of concerns (config, services, state)
  * Easily extensible for new models and pipelines

---

## 🧠 Architecture Overview

```text
User Query / PDF Upload
        │
        ▼
Streamlit UI (Frontend)
        │
        ▼
Document Processing
(Chunking + FastEmbed)
        │
        ▼
Qdrant Vector Store
(Embedding Storage)
        │
        ▼
Retriever (Top-K Search)
        │
        ▼
LLM (OpenAI Agent)
        │
        ▼
Response Refinement
(Speech Optimization)
        │
        ▼
Text-to-Speech (TTS)
        │
        ▼
Audio Output (Playback / Download)
```

---

## 📂 Project Structure

```text
.
├── app.py                  # Streamlit entrypoint
├── pyproject.toml
├── requirements.txt
├── src/
│   └── voice_ai_agent/
│       ├── app.py          # UI logic
│       ├── config.py       # Configuration management
│       ├── services.py     # Core pipeline (RAG + TTS)
│       └── state.py        # Session state handling
└── tests/
    └── test_config.py
```

---

## ⚡ Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
```

### 3. Run the Application

```bash
streamlit run app.py
```

---

## 🔄 How It Works

1. PDFs are uploaded and split into meaningful chunks
2. Chunks are embedded using FastEmbed
3. Embeddings are stored in Qdrant
4. Relevant chunks are retrieved based on the query
5. LLM generates a context-aware answer
6. Response is optimized for speech clarity
7. TTS converts text into audio for playback

---

## 🎥 Demo

### Example Query

> "Summarize the key points from the uploaded document"

### Output

* Context-aware response generated using RAG
* Audio playback using TTS

*(Add a short demo GIF or screen recording here for maximum impact)*

---

## ⚙️ Configuration (Upcoming)

```python
mode = "openai"   # or "ollama"
tts  = "openai"   # or "gtts"
```

Supports:

* Cloud inference (OpenAI)
* Local inference (Ollama)
* Paid TTS (OpenAI)
* Free TTS fallback (gTTS)

---

## 🔮 Roadmap

### 🔓 Local / Offline Mode

* Ollama integration for local LLM inference
* No API key required
* Fully offline document Q&A
* Improved privacy

### 🔊 Free TTS Pipeline

* gTTS fallback integration
* Automatic playback (no download needed)
* Lower latency and zero cost

### ⚙️ Smart Routing

* Automatic fallback:

  * OpenAI → Ollama
  * OpenAI TTS → gTTS
* Config-driven execution

### 🚀 Performance Improvements

* Streaming responses (text + audio)
* Embedding and query caching
* Adaptive chunking strategies

### 📊 Evaluation & Observability

* RAG evaluation metrics (precision, recall)
* Logging and tracing
* Pipeline monitoring dashboard

---

## 🧪 Development

```bash
make install
make run
make test
```

---

## ⚠️ Limitations

* Depends on external APIs (OpenAI, Qdrant) in current setup
* No streaming support yet
* Limited evaluation tooling

---

## 🏷️ Keywords

`rag` `llm` `voice-ai` `qdrant` `streamlit` `openai` `tts` `ai-agents` `ollama`

---

## 📌 Positioning

This project is designed as a **foundation for production-grade voice-enabled RAG systems**, with extensibility toward:

* Local LLM deployment (Ollama)
* Cost-optimized AI pipelines
* Real-time conversational interfaces

---
