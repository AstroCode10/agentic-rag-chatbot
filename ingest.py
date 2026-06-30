from chunking import chunk_text
from embed_store import embed_and_store

# Paths of all relevant documents to be chunked and embedded
docs = {
    "vela_orders_and_billing" : "docs/vela_orders_and_billing.txt",
    "vela_pricing_and_plans" : "docs/vela_pricing_and_plans.txt",
    "vela_product_overview" : "docs/vela_product_overview.txt",
    "vela_troubleshooting" : "docs/vela_troubleshooting.txt"
}

# Initialize an empty list to hold all chunks from all documents
all_chunks = []

# FOR loop to chunk all documents
for doc in docs.keys():
    with open(docs[doc], "r") as f:
        text = f.read()
        chunks = chunk_text(text)

        for chunk in chunks:
            chunk["source"] = doc
        
        all_chunks.extend(chunks)

# Embedding and storing all documents
embed_and_store(all_chunks, collection_name="vela_support")