from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")


def build_faiss_index(chunks, index_path="faiss_index"):
    print("Embedding Chunks")

    embeddings = model.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]  # 384
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    if not os.path.exists(index_path):
        os.makedirs(index_path)

    faiss.write_index(index, os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved faiss index {len(chunks)} chunks")


def search_index(query, top_k = 5, index_path="faiss_index"):
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    query_vec = model.encode([query])
    query_vec = np.array(query_vec)

    distances, indices = index.search(query_vec, top_k)

    results = [chunks[i] for i in indices[0]]
    return results


if __name__ == "__main__":
    from pdf_parser import extract_text_from_pdf, chunk_text
    text = extract_text_from_pdf("example.pdf")
    chunks = chunk_text(text)
    build_faiss_index(chunks)

    while True:
        q = input("ask question")
        if q.lower() == "exit":
            break
        results = search_index(q)
        print("\n".join(results[:3]))
        print("-" * 80)