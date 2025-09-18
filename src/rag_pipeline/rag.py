from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import EmbeddingFunction
import chromadb

class CustomEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using SentenceTransformers."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input_texts):
        return self.model.encode(input_texts).tolist()

class RAGPipeline:
    """RAG pipeline with custom embeddings and metadata-aware retrieval."""
    def __init__(self, collection_name="documents", embedding_model="all-MiniLM-L6-v2"):
        self.embedding_func = CustomEmbeddingFunction(model_name=embedding_model)
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_func
        )

    def add_document(self, doc_id: str, summary: str, doc_type: str):
        """Add a single document with metadata."""
        self.collection.add(
            documents=[summary],
            metadatas=[{"doc_type": doc_type}],
            ids=[doc_id]
        )

    def add_documents(self, docs: list):
        """
        Add multiple documents.
        Each doc is a dict: {"id": ..., "summary": ..., "doc_type": ...}
        """
        for d in docs:
            self.add_document(d["id"], d["summary"], d["doc_type"])

    def query(self, query_text: str, n_results=3):
        """Query the RAG system."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        retrieved_docs = results["documents"][0]
        retrieved_types = [m["doc_type"] for m in results["metadatas"][0]]
        return list(zip(retrieved_types, retrieved_docs))
