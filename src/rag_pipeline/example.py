from rag import RAGPipeline

# Initialize RAG pipeline
rag = RAGPipeline()

# Add multiple documents
docs = [
    {"id": "1", "summary": "Quarterly financial report showing revenue growth.", "doc_type": "Financial Report"},
    {"id": "2", "summary": "Invoice for software subscription services.", "doc_type": "Invoice"},
    {"id": "3", "summary": "Company HR policy regarding remote work.", "doc_type": "Policy Document"},
    {"id": "4", "summary": "Technical documentation for API usage.", "doc_type": "Tech Doc"}
]

rag.add_documents(docs)

# Query the system
query = "rules about working from home"
results = rag.query(query, n_results=2)

# Print retrieved summaries and document types
for doc_type, summary in results:
    print("Doc Type:", doc_type)
    print("Summary:", summary)
    print("------")
