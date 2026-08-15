from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

TOP_K = 4

def retrieve_relevant_documents(vector_store: InMemoryVectorStore, query: str) -> list[Document]:
    return vector_store.similarity_search(query, k=TOP_K)