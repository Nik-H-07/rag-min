import os
from pathlib import Path
import json
import streamlit as st
from rich import print  # nur für Konsolenlogs (optional)
from config import CFG

# Funktionen aus deinem bestehenden Code wiederverwenden
from ingest import load_docs, chunk_pages, build_index
from ask import load_store, search, answer_with_llm

DATA_DIR = Path("data/raw")
INDEX_DIR = Path("index")

st.set_page_config(page_title="RAG • Q&A", page_icon="🔎", layout="wide")

st.title("🔎 RAG – Fragen an eure Dokumente")
st.caption("FAISS + Sentence-Transformers + OpenAI/Ollama (OpenAI-kompatibel)")

# --- Sidebar: Einstellungen / Index-Management ---
with st.sidebar:
    st.header("⚙️ Einstellungen")
    chunk_size = st.number_input("Chunk Size (Tokens ≈ Zeichen)", 200, 4000, CFG["CHUNK_SIZE"], step=50)
    chunk_overlap = st.number_input("Chunk Overlap", 0, 1000, CFG["CHUNK_OVERLAP"], step=10)
    top_k = st.slider("Top-K Kontexte", 1, 20, CFG["TOP_K"])
    use_reranker = st.checkbox("Re-Ranking aktivieren (präziser, langsamer)", value=CFG["USE_RERANKER"])

    st.divider()
    st.header("📄 Dokumente hochladen")
    uploaded = st.file_uploader("PDF/DOCX/TXT (mehrfach möglich)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for file in uploaded:
            out = DATA_DIR / file.name
            with open(out, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"{len(uploaded)} Datei(en) gespeichert in {DATA_DIR}")

    st.divider()
    rebuild = st.button("🧱 Index neu bauen")

# --- Hauptbereich: Index bauen, Fragen stellen ---
tab_ask, tab_passages = st.tabs(["💬 Fragen", "📚 Gefundene Passagen"])

# Index bauen
if rebuild:
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        st.error("Kein Input gefunden. Bitte zuerst Dateien in `data/raw` hochladen.")
    else:
        with st.status("Baue Index …", expanded=True) as status:
            st.write("📥 Lade & parse Dokumente …")
            pages = load_docs(DATA_DIR)
            st.write(f"✂️ Chunking ({len(pages)} Seiten) …")
            chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.write(f"🧮 Embeddings & FAISS-Index ({len(chunks)} Chunks) …")
            build_index(chunks, CFG["EMBEDDING_MODEL"], INDEX_DIR)
            status.update(label="Fertig ✅", state="complete", expanded=False)
        st.success("Index erstellt. Du kannst jetzt Fragen stellen.")

with tab_ask:
    st.subheader("Stelle eine Frage an eure Dokumente")
    question = st.text_area("Frage", placeholder="Z. B. Welche Vorteile hat Hybrid-Retrieval?")
    ask_btn = st.button("Antwort erzeugen", type="primary")

    if ask_btn:
        if not INDEX_DIR.exists() or not (INDEX_DIR / "index.faiss").exists():
            st.error("Kein Index gefunden. Bitte zuerst rechts in der Sidebar den Index bauen.")
        elif not question.strip():
            st.warning("Bitte gib eine Frage ein.")
        else:
            try:
                index, store, settings = load_store(INDEX_DIR)
                passages = search(
                    question, index, store, settings["embedding_model"],
                    top_k=top_k, rerank=use_reranker
                )
                if not passages:
                    st.info("Keine passenden Passagen gefunden.")
                else:
                    # Antwort vom LLM
                    with st.spinner("Frage Modell …"):
                        answer = answer_with_llm(question, passages)

                    st.markdown("### ✨ Antwort")
                    st.write(answer)

                    # Speichere gefundene Passagen in Session für Tab 2
                    st.session_state["last_passages"] = passages
            except Exception as e:
                st.error(f"Fehler: {e}")

with tab_passages:
    st.subheader("Top-Passagen aus dem Index")
    passages = st.session_state.get("last_passages", [])
    if not passages:
        st.info("Noch keine Passagen – stelle zuerst eine Frage.")
    else:
        for i, p in enumerate(passages, start=1):
            meta = p.get("meta", {})
            src = os.path.basename(meta.get("source") or "Unbekannt")
            page = f" (Seite {meta.get('page')})" if meta.get("page") else ""
            with st.expander(f"[{i}] {src}{page}"):
                st.write(p["text"])
