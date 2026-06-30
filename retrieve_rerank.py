import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# Retrieval
def retrieve(query:str, collection: chromadb.Collection, model: SentenceTransformer,
             top_k: int = 20) -> list[dict]:
    
    # Embed the query using the embedding model
    query_embeddings = model.encode(query,
                                    show_progress_bar=True,
                                    batch_size=32,
                                    normalize_embeddings=True).to_list()
    results = collection.query(
        query_embeddings=[query_embeddings],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
        )

    documents = results["documents"][0] # List of chunk texts
    metadatas = results["metadatas"][0] # List of metadata dictionaries for each chunk
    distances = results["distances"][0] # List of distances for each chunk
    ids = results["ids"][0] # List of ids for each chunk

    candidates = []

    for doc, meta, dist, id in zip(documents, metadatas, distances, ids):
        candidates.append({
            "text": doc,
            "id": id,
            "cosine_dist": dist,
            "cosine_sim": 1 - dist,
            "metadata": meta,
        })
    
    return candidates
    

# Reranking
def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    
    # Initialize the CrossEncoder model for reranking
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = cross_encoder.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = score
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return candidates[:top_n]

# Combined function for retrieval and reranking
def retrieve_and_rerank(query: str, collection: chromadb.Collection, model: SentenceTransformer,
                        initial_k: int = 20, final_k: int = 5) -> list[dict]:
    candidates = retrieve(query, collection, model, top_k=initial_k)
    top_chunks = rerank(query, candidates, top_n=final_k)
    return top_chunks

# Format context
def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"Source{i}\n{chunk['text']}")
    return "\n\n----\n\n".join(parts)