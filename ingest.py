from langchain_community.document_loaders import PyPDFDirectoryLoader

def ingest_documents():
    loader = PyPDFDirectoryLoader('./knowledge_base')
    documents = loader.load()
    return documents