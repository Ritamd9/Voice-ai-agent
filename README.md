# Voice AI Agent

A production-ready Streamlit application for document question-answering with voice playback. The app indexes PDF content into Qdrant, retrieves the most relevant chunks for a user query, generates an answer with the OpenAI Agents SDK, and returns an MP3 response using OpenAI text-to-speech.

## Features

- PDF ingestion with chunking and metadata enrichment
- Semantic retrieval backed by Qdrant and FastEmbed
- Two-step answer generation for response quality and TTS optimization
- Browser-based audio playback and downloadable MP3 responses
- `src/`-based Python package layout with lightweight tests

## Project Structure

```text
.
├── app.py
├── pyproject.toml
├── requirements.txt
├── src/
│   └── voice_ai_agent/
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       ├── services.py
│       └── state.py
└── tests/
    └── test_config.py
```

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your credentials:

```bash
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
```

4. Run the app:

```bash
streamlit run app.py
```

## How It Works

1. Uploaded PDFs are split into chunks and embedded with FastEmbed.
2. Embeddings are stored in a Qdrant collection.
3. The most relevant chunks are retrieved for each question.
4. An OpenAI agent drafts the answer and a second pass optimizes it for speech.
5. OpenAI TTS generates an MP3 file that can be played and downloaded from the UI.

## Development

- `make install` installs the application dependencies.
- `make run` launches the Streamlit app locally.
- `make test` runs the lightweight unit test suite.

