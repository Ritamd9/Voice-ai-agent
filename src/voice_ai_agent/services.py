from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
import os
import tempfile
import uuid

from agents import Agent, Runner
from fastembed import TextEmbedding
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from voice_ai_agent.config import AppConfig


ANSWER_MODEL = "gpt-4o"
TTS_INSTRUCTION_MODEL = "gpt-4o-mini"
SPEECH_MODEL = "gpt-4o-mini-tts"
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200
SEARCH_LIMIT = 3


def setup_qdrant(config: AppConfig) -> tuple[QdrantClient, TextEmbedding]:
    if not config.qdrant_url or not config.qdrant_api_key:
        raise ValueError("Qdrant credentials are required before indexing documents.")

    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    embedding_model = TextEmbedding()
    embedding_dim = len(list(embedding_model.embed(["healthcheck"]))[0])

    try:
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
    except Exception as exc:
        if "already exists" not in str(exc):
            raise

    return client, embedding_model


def process_pdf(uploaded_file: Any) -> list[Any]:
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = Path(temp_file.name)

        loader = PyPDFLoader(str(temp_path))
        documents = loader.load()

        for document in documents:
            document.metadata.update(
                {
                    "source_type": "pdf",
                    "file_name": uploaded_file.name,
                    "indexed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        return splitter.split_documents(documents)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def store_embeddings(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    documents: Sequence[Any],
    collection_name: str,
) -> None:
    if not documents:
        return

    contents = [document.page_content for document in documents]
    embeddings = list(embedding_model.embed(contents))
    points = []

    for document, embedding in zip(documents, embeddings, strict=True):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "content": document.page_content,
                    **document.metadata,
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)


def setup_agents(openai_api_key: str) -> tuple[Agent, Agent]:
    if not openai_api_key:
        raise ValueError("OpenAI API key is required before processing queries.")

    os.environ["OPENAI_API_KEY"] = openai_api_key

    processor_agent = Agent(
        name="Documentation Processor",
        instructions=(
            "You answer questions using retrieved documentation context only. "
            "Be concise, accurate, and cite the source file names when relevant. "
            "If the context is insufficient, say so explicitly."
        ),
        model=ANSWER_MODEL,
    )

    tts_agent = Agent(
        name="Speech Formatter",
        instructions=(
            "Rewrite the answer so it sounds natural when spoken aloud. "
            "Keep the meaning unchanged, preserve technical accuracy, and avoid markdown."
        ),
        model=TTS_INSTRUCTION_MODEL,
    )

    return processor_agent, tts_agent


def search_documents(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
    collection_name: str,
    limit: int = SEARCH_LIMIT,
) -> Sequence[Any]:
    query_embedding = list(embedding_model.embed([query]))[0]
    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=limit,
        with_payload=True,
    )
    return response.points if hasattr(response, "points") else []


def build_context(search_results: Iterable[Any], query: str) -> tuple[str, list[str]]:
    sections: list[str] = []
    sources: list[str] = []

    for result in search_results:
        payload = result.payload or {}
        content = str(payload.get("content", "")).strip()
        source = str(payload.get("file_name", "Unknown Source"))

        if not content:
            continue

        sections.append(f"Source: {source}\nExcerpt:\n{content}")
        if source not in sources:
            sources.append(source)

    if not sections:
        raise ValueError("No relevant document content was retrieved from Qdrant.")

    context = (
        "Use the documentation excerpts below to answer the user question.\n\n"
        + "\n\n".join(sections)
        + f"\n\nUser question: {query}"
    )

    return context, sources


async def generate_audio_file(
    text_response: str,
    voice_instructions: str,
    voice: str,
    openai_api_key: str,
) -> str:
    client = AsyncOpenAI(api_key=openai_api_key)
    audio_response = await client.audio.speech.create(
        model=SPEECH_MODEL,
        voice=voice,
        input=text_response,
        instructions=voice_instructions,
        response_format="mp3",
    )

    output_path = Path(tempfile.gettempdir()) / f"voice_ai_agent_{uuid.uuid4()}.mp3"
    output_path.write_bytes(audio_response.content)
    return str(output_path)


async def process_query(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
    collection_name: str,
    openai_api_key: str,
    voice: str,
    processor_agent: Agent,
    tts_agent: Agent,
) -> dict[str, Any]:
    search_results = search_documents(
        query=query,
        client=client,
        embedding_model=embedding_model,
        collection_name=collection_name,
    )
    context, sources = build_context(search_results, query)

    processor_result = await Runner.run(processor_agent, context)
    text_response = processor_result.final_output

    tts_result = await Runner.run(tts_agent, text_response)
    voice_instructions = tts_result.final_output

    audio_path = await generate_audio_file(
        text_response=text_response,
        voice_instructions=voice_instructions,
        voice=voice,
        openai_api_key=openai_api_key,
    )

    return {
        "text_response": text_response,
        "voice_instructions": voice_instructions,
        "audio_path": audio_path,
        "sources": sources,
    }

