import os, json, argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
import docx2txt
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from rich import print
from config import CFG

def read_pdf(fp: Path) -> List[Dict]:
    reader = PdfReader(str(fp))
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append({"text": txt, "page": i + 1})
    return pages

def read_txt(fp: Path) -> List[Dict]:
    txt = fp.read_text(encoding="utf-8", errors="ignore")
    return [{"text": txt, "page": None}]

def read_docx(fp: Path) -> List[Dict]:
    txt = docx2txt.process(str(fp)) or ""
    return [{"text": txt, "page": None}]

def load_docs(data_dir: Path) -> List[Dict]:
    docs = []
    for fp in sorted(data_dir.rglob("*")):
        if not fp.is_file(): 
            continue
        if fp.suffix.lower() == ".pdf":
            for p in read_pdf(fp):
                docs.append({"source": str(fp), **p})
        elif fp.suffix.lower() in {".txt", ".md"}:
            for p in read_txt(fp):
                p["source"] = str(fp)
                docs.append(p)
        elif fp.suffix.lower() in {".docx"}:
            for p in read_docx(fp):
                p["source"] = str(fp)
                docs.append(p)
    return docs

def chunk_pages(pages: List[Dict], chunk_size=800, chunk_overlap=120) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        add_start_index=True, separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for item in pages:
        for ch in splitter.split_text(item["text"]):
            chunks.append({
                "text": ch,
                "source": item.get("source"),
                "page": item.get("page"),
            })
    return chunks

def build_index(chunks: List[Dict], embed_model: str, index_dir: Path):
    model = SentenceTransformer(embed_model)
    texts = [c["text"] for c in chunks]
    print(f"[green]Embedding {len(texts)} Chunks mit {embed_model}...[/green]")
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine via normalisierte Vektoren
    index.add(embs)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    # Texte + Metadaten
    with open(index_dir / "store.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            row = {"text": c["text"], "meta": {"source": c["source"], "page": c["page"]}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(index_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump({"embedding_model": embed_model}, f)

    print(f"[bold green]Index gespeichert in: {index_dir}[/bold green]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/raw")
    ap.add_argument("--index", type=str, default="index")
    ap.add_argument("--chunk-size", type=int, default=CFG["CHUNK_SIZE"])
    ap.add_argument("--overlap", type=int, default=CFG["CHUNK_OVERLAP"])
    ap.add_argument("--embedding-model", type=str, default=CFG["EMBEDDING_MODEL"])
    args = ap.parse_args()

    data_dir = Path(args.data)
    index_dir = Path(args.index)
    assert data_dir.exists(), f"Ordner {data_dir} nicht gefunden."

    print("[cyan]Lese Dokumente...[/cyan]")
    pages = load_docs(data_dir)
    if not pages:
        print("[red]Keine Dokumente gefunden (PDF/TXT/DOCX).[/red]")
        return

    print(f"[cyan]Chunking {len(pages)} Seiten...[/cyan]")
    chunks = chunk_pages(pages, args.chunk_size, args.overlap)
    print(f"[cyan]{len(chunks)} Chunks erzeugt.[/cyan]")

    build_index(chunks, args.embedding_model, index_dir)

if __name__ == "__main__":
    main()
