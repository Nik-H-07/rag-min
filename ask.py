import os, json, argparse
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rich import print
from openai import OpenAI
from config import CFG

def load_store(index_dir: Path):
    index = faiss.read_index(str(index_dir / "index.faiss"))
    store = []
    with open(index_dir / "store.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            store.append(json.loads(line))
    with open(index_dir / "settings.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
    return index, store, settings

def search(query: str, index, store, embed_model: str, top_k=6, rerank=False):
    embedder = SentenceTransformer(embed_model)
    q = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(q, dtype="float32"), top_k * 8 if rerank else top_k)
    cand = [{"text": store[i]["text"], "meta": store[i]["meta"], "score": float(D[0][j])} 
            for j, i in enumerate(I[0]) if i != -1]

    if rerank:
        try:
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, c["text"]) for c in cand]
            scores = reranker.predict(pairs).tolist()
            for c, s in zip(cand, scores):
                c["rerank"] = float(s)
            cand.sort(key=lambda x: x.get("rerank", 0), reverse=True)
            cand = cand[:top_k]
        except Exception as e:
            print(f"[yellow]Re-Ranking nicht verfügbar ({e}); verwende Rohtreffer.[/yellow]")
            cand = cand[:top_k]
    else:
        cand = cand[:top_k]
    return cand

def build_prompt(query: str, passages):
    ctx_blocks, src_list = [], []
    for idx, p in enumerate(passages, start=1):
        src = p["meta"]
        s = os.path.basename(src["source"] or "Unbekannt")
        page = f" (Seite {src['page']})" if src.get("page") else ""
        label = f"[{idx}] {s}{page}"
        ctx_blocks.append(f"[{idx}] {p['text']}")
        src_list.append(label)
    context = "\n\n".join(ctx_blocks)
    sources = "\n".join(src_list)

    system = (
        "Du bist ein präziser Assistent für dokumentenbasiertes Fragenbeantworten. "
        "Beantworte NUR mit den bereitgestellten Kontextpassagen. "
        "Jede faktische Aussage muss mit Quellenhinweisen im Format [1], [2], ... belegt sein. "
        "Wenn die Antwort im Kontext nicht ausreichend belegt ist, sage 'Unklare Datenlage' "
        "und zeige die relevantesten Quellen."
    )
    user = f"FRAGE:\n{query}\n\nKONTEXT:\n{context}\n\nQUELLEN:\n{sources}\n"
    return system, user

def answer_with_llm(query: str, passages):
    base_url = CFG["OPENAI_BASE_URL"]
    client = OpenAI(api_key=CFG["OPENAI_API_KEY"], base_url=base_url) if base_url \
             else OpenAI(api_key=CFG["OPENAI_API_KEY"])
    system, user = build_prompt(query, passages)
    resp = client.chat.completions.create(
        model=CFG["LLM_MODEL"],
        temperature=0.2,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
    )
    return resp.choices[0].message.content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str, help="Deine Frage in natürlicher Sprache.")
    ap.add_argument("--index", type=str, default="index")
    ap.add_argument("--k", type=int, default=CFG["TOP_K"])
    ap.add_argument("--rerank", action="store_true" if CFG["USE_RERANKER"] else "store_false")
    args = ap.parse_args()

    index_dir = Path(args.index)
    assert index_dir.exists(), f"Index {index_dir} nicht gefunden. Erst ingest.py ausführen."

    index, store, settings = load_store(index_dir)
    passages = search(args.question, index, store, settings["embedding_model"], top_k=args.k,
                      rerank=args.rerank or CFG["USE_RERANKER"])

    if not passages:
        print("[red]Keine passenden Passagen gefunden.[/red]")
        return

    print(f"[cyan]Top-{len(passages)} Kontexte geladen. Erzeuge Antwort...[/cyan]")
    out = answer_with_llm(args.question, passages)
    print("\n[bold]Antwort:[/bold]\n")
    print(out)

if __name__ == "__main__":
    main()
