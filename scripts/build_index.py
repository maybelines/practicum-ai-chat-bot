import argparse
import json
from pathlib import Path
from time import perf_counter

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "vector_index"
MODEL_NAME = "intfloat/multilingual-e5-base"


def load_model():
    torch.set_num_threads(4)
    return SentenceTransformer(MODEL_NAME, device="cpu", cache_folder=str(ROOT / ".cache/models"))


def build_index(model):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        length_function=lambda text: len(text.split()),
        separators=[r"(?<=[.!?])\s+", r"\s+", ""],
        is_separator_regex=True,
    )
    chunks = []
    for path in sorted((ROOT / "knowledge_base").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ")
        start = -1
        for part in splitter.split_text(text):
            start = text.index(part, start + 1)
            chunks.append({
                "id": len(chunks),
                "source": path.relative_to(ROOT).as_posix(),
                "title": title,
                "start_char": start,
                "end_char": start + len(part),
                "text": part,
            })
    if not chunks:
        raise ValueError("No documents in knowledge_base/")

    passages = [f"passage: {chunk['title']}\n{chunk['text']}" for chunk in chunks]
    for passage in passages:
        if len(model.tokenizer.encode(passage)) > model.max_seq_length:
            raise ValueError("Chunk is too long for the model; reduce chunk_size")
    started = perf_counter()
    vectors = model.encode(passages, batch_size=8, normalize_embeddings=True)
    seconds = round(perf_counter() - started, 2)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    data = {"model": MODEL_NAME, "embedding_seconds": seconds, "chunks": chunks}
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {index.ntotal} chunks, dimension {index.d}, embeddings: {seconds} s")


def search(model, query):
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    data = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    if data["model"] != MODEL_NAME or len(data["chunks"]) != index.ntotal:
        raise ValueError("Rebuild the index: model or chunk count does not match")
    vector = model.encode([f"query: {query}"], normalize_embeddings=True)
    scores, ids = index.search(vector, min(3, index.ntotal))
    return [(data["chunks"][int(i)], float(score)) for i, score in zip(ids[0], scores[0])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS index or search it")
    parser.add_argument("query", nargs="?", help="Search query; omit to build the index")
    args = parser.parse_args()
    model = load_model()
    if args.query:
        for chunk, score in search(model, args.query):
            print(f"\n{chunk['title']} — {chunk['source']}, chunk {chunk['id']}, "
                  f"chars {chunk['start_char']}:{chunk['end_char']}, score {score:.3f}")
            print(chunk["text"])
    else:
        build_index(model)
