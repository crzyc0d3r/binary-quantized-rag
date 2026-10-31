"""Local Streamlit front end for native binary retrieval and generation."""
import hashlib
from pathlib import Path
import tempfile
import streamlit as st
from rag import NativePipeline, answer

st.title("Binary Quantized RAG")
st.write("Upload a text, Markdown, or PDF document, then retrieve packed binary vectors and ask questions.")
uploaded = st.file_uploader("Document", type=["txt", "md", "pdf"])
if uploaded and st.button("Build document index"):
    data = uploaded.getvalue()
    if not data or len(data) > 20 * 1024 * 1024:
        st.error("Choose a nonempty document smaller than 20 MiB.")
    else:
        try:
            with tempfile.TemporaryDirectory() as directory:
                # Never use an uploaded filename as an arbitrary filesystem path.
                destination = Path(directory) / ("document" + Path(uploaded.name).suffix.lower())
                destination.write_bytes(data)
                with st.spinner("Extracting, embedding and indexing"):
                    pipeline = NativePipeline(directory)
            previous = st.session_state.get("pipeline")
            if previous:
                previous.close()
            st.session_state.pipeline = pipeline
            st.session_state.digest = hashlib.sha256(data).hexdigest()
            st.session_state.collection = pipeline.index.collection
            st.success("Index ready.")
        except Exception as exc:
            st.error(f"Indexing failed ({type(exc).__name__}). Check model access and Milvus configuration.")

pipeline = st.session_state.get("pipeline")
if pipeline:
    st.caption(f"Collection: {pipeline.index.collection}")
    st.write(f"Vector payload: {pipeline.payload_bytes:,} packed bytes versus "
             f"{pipeline.float32_bytes:,} float32 bytes (32×). Database overhead is additional.")
    question = st.text_input("Question")
    if st.button("Retrieve and answer", disabled=not question.strip()):
        if uploaded and hashlib.sha256(uploaded.getvalue()).hexdigest() != st.session_state.digest:
            st.error("The selected document changed. Build its index before querying it.")
        else:
            try:
                hits = pipeline.retrieve(question)
                for i, hit in enumerate(hits, 1):
                    st.write(f"[{i}] {hit['source']} · Hamming distance {hit['distance']}")
                    st.text(hit["text"])
                st.text(answer(question, hits))
            except Exception as exc:
                st.error(f"Query failed ({type(exc).__name__}). Check the provider key and model configuration.")
