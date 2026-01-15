from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def lexical_retriever(documents, k):
    return BM25Retriever.from_documents(documents, k=k)


def semantic_retriever(vectorstore, k):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def hybrid_retriever(vectorstore, documents, weights=[0.38, 0.62], k=10):
    bm25 = lexical_retriever(documents, k=k)
    vector = semantic_retriever(vectorstore, k=k)
    return EnsembleRetriever(retrievers=[bm25, vector], weights=weights)
