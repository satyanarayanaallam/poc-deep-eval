import chromadb
from chromadb.utils import embedding_functions
from embedding import GeminiEmbeddingFunction

# Use OpenAI embeddings (replace with Gemini/OpenRouter if you want)
embedding_func = GeminiEmbeddingFunction(model="models/embedding-001")

# Initialize Chroma client
client = chromadb.Client()

# Create / get collection
collection = client.create_collection(name="documents", embedding_function=embedding_func)

# Sample documents
docs = [
    {"id": "1", "summary": "Quarterly financial report showing revenue growth.", "doc_type": "Financial Report"},
    {"id": "2", "summary": "Invoice for software subscription services.", "doc_type": "Invoice"},
    {"id": "3", "summary": "Company HR policy regarding remote work.", "doc_type": "Policy Document"}
]

# Ingest into Chroma
for d in docs:
    collection.add(
        documents=[d["summary"]],
        metadatas=[{"doc_type": d["doc_type"]}],
        ids=[d["id"]]
    )

# Query
query = "remote work rules for employees"
results = collection.query(
    query_texts=[query],
    n_results=2
)

# Print results
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print("Summary:", doc)
    print("Doc Type:", meta["doc_type"])
    print("----")
