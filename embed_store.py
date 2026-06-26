import chromadb
from sentence_transformers import SentenceTransformer

def embed_and_store(chunks: list[dict], collection_name: str = "docs") -> chromadb.Collection:

    # Initialize the SentenceTransformer model and ChromaDB client
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient()

    # Getting the essential data to add to the collection
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    text = [chunk["text"] for chunk in chunks]
    ids = [f"chunk_{(chunk["index"])}" for chunk in chunks]
    embeddings = model.encode(text, show_progress_bar=True, batch_size=32, normalize_embeddings=True).to_list()
    
    # Add the embeddings to the collection with metadata
    collection.add(
        ids=ids,
        documents=text,
        embeddings=embeddings,
        metadatas=[{"source": "chatbot", "chunk_index": chunk["index"]} for chunk in chunks],
        )
    
    return collection
