from __future__ import annotations

import asyncio

import streamlit as st

from voice_ai_agent.config import AppConfig, SUPPORTED_VOICES
from voice_ai_agent.services import process_pdf, process_query, setup_agents, setup_qdrant, store_embeddings
from voice_ai_agent.state import initialize_session_state


def render_sidebar() -> AppConfig:
    with st.sidebar:
        st.header("Configuration")
        st.caption("Environment values from `.env` are loaded automatically and can be overridden here.")

        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
        )
        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url,
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value=st.session_state.qdrant_api_key,
            type="password",
        )
        st.session_state.selected_voice = st.selectbox(
            "Voice",
            options=SUPPORTED_VOICES,
            index=SUPPORTED_VOICES.index(st.session_state.selected_voice),
        )

        config = AppConfig.from_state(st.session_state)
        if config.is_complete:
            st.success("Credentials are configured.")
        else:
            missing = ", ".join(config.missing_fields())
            st.warning(f"Missing: {missing}")

        if st.session_state.processed_documents:
            st.divider()
            st.subheader("Indexed Documents")
            for file_name in st.session_state.processed_documents:
                st.write(f"- {file_name}")

    return config


def ensure_runtime_dependencies(config: AppConfig) -> None:
    if st.session_state.client is None or st.session_state.embedding_model is None:
        client, embedding_model = setup_qdrant(config)
        st.session_state.client = client
        st.session_state.embedding_model = embedding_model

    if st.session_state.processor_agent is None or st.session_state.tts_agent is None:
        processor_agent, tts_agent = setup_agents(config.openai_api_key)
        st.session_state.processor_agent = processor_agent
        st.session_state.tts_agent = tts_agent


def render_header() -> None:
    st.title("Voice AI Agent")
    st.write(
        "Upload PDF documents, ask questions about their contents, and receive an answer with generated voice playback."
    )


def render_upload_section(config: AppConfig) -> None:
    st.subheader("1. Index Documents")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if not uploaded_file:
        return

    if uploaded_file.name in st.session_state.processed_documents:
        st.info("This document is already indexed in the current session.")
        return

    if not config.is_complete:
        st.warning("Add your OpenAI and Qdrant credentials in the sidebar before indexing documents.")
        return

    with st.spinner(f"Indexing {uploaded_file.name}..."):
        try:
            ensure_runtime_dependencies(config)
            documents = process_pdf(uploaded_file)
            store_embeddings(
                client=st.session_state.client,
                embedding_model=st.session_state.embedding_model,
                documents=documents,
                collection_name=config.collection_name,
            )
            st.session_state.processed_documents.append(uploaded_file.name)
            st.success(f"{uploaded_file.name} indexed successfully.")
        except Exception as exc:
            st.error(f"Failed to index document: {exc}")


def render_query_section(config: AppConfig) -> None:
    st.subheader("2. Ask Questions")
    indexing_ready = bool(st.session_state.processed_documents)

    with st.form("query-form", clear_on_submit=False):
        query = st.text_input(
            "Ask a question about the indexed documents",
            placeholder="What are the main ideas in this PDF?",
            disabled=not indexing_ready,
        )
        submitted = st.form_submit_button("Generate Answer", disabled=not indexing_ready)

    if not indexing_ready:
        st.info("Index at least one PDF to unlock querying.")
        return

    if not submitted or not query.strip():
        return

    try:
        ensure_runtime_dependencies(config)
        with st.spinner("Searching documents and generating audio response..."):
            result = asyncio.run(
                process_query(
                    query=query.strip(),
                    client=st.session_state.client,
                    embedding_model=st.session_state.embedding_model,
                    collection_name=config.collection_name,
                    openai_api_key=config.openai_api_key,
                    voice=st.session_state.selected_voice,
                    processor_agent=st.session_state.processor_agent,
                    tts_agent=st.session_state.tts_agent,
                )
            )
    except Exception as exc:
        st.error(f"Failed to process query: {exc}")
        return

    st.markdown("### Answer")
    st.write(result["text_response"])

    st.markdown("### Audio")
    st.audio(result["audio_path"], format="audio/mp3")
    with open(result["audio_path"], "rb") as audio_file:
        st.download_button(
            label="Download MP3",
            data=audio_file.read(),
            file_name=f"voice-response-{st.session_state.selected_voice}.mp3",
            mime="audio/mp3",
        )

    st.markdown("### Sources")
    for source in result["sources"]:
        st.write(f"- {source}")


def main() -> None:
    st.set_page_config(page_title="Voice AI Agent", page_icon="🎙️", layout="wide")
    initialize_session_state(st.session_state)

    config = render_sidebar()
    render_header()
    render_upload_section(config)
    st.divider()
    render_query_section(config)


if __name__ == "__main__":
    main()

