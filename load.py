import hashlib
from clients import get_chroma_vectorstore, get_gemini_embeddings
from ingest import ingest_documents
from transform import transform_documents_chunksize


def load_data():
    documents = ingest_documents()
    embeddings = get_gemini_embeddings()
    documents_chunks = transform_documents_chunksize(documents)
    vectorstore = get_chroma_vectorstore(embeddings)
    add_documents_without_duplicates(documents_chunks, vectorstore)
    return vectorstore, documents_chunks


def add_documents_without_duplicates(documents, vectorstore):
    unique_docs_map = {}
    for i, doc in enumerate(documents):
        # Production-grade ID generation:
        # We combine content + source + page + chunk_index to ensure uniqueness across files and positions
        source = doc.metadata.get("source", "unknown")
        # Handle cases where page metadata might be missing or different types
        page = doc.metadata.get("page", "no_page")

        # We include 'i' (chunk index) or a chunk identifier if available to distinguish
        # identical text appearing twice in the same general context (rare, but possible)
        # However, for pure deduplication, we might WANT to skip identical text in the same file/page.
        # Let's settle on: Source + Page + Content.
        # This allows the same text in a different file (or page) to be treated as a new record.
        string_to_hash = f"{source}:{page}:{doc.page_content}"

        doc_id = hashlib.sha256(string_to_hash.encode()).hexdigest()

        if doc_id not in unique_docs_map:
            unique_docs_map[doc_id] = doc

    # IDs to check in the vectorstore (now guaranteed to be unique)
    doc_ids = list(unique_docs_map.keys())

    if not doc_ids:
        return []

    # Check which of these already exist in Chroma
    existing_results = vectorstore.get(ids=doc_ids)
    existing_ids = set(existing_results["ids"])

    new_docs = []
    new_ids = []

    for doc_id, doc in unique_docs_map.items():
        if doc_id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)

    if new_docs:
        vectorstore.add_documents(new_docs, ids=new_ids)
        print(f"Added {len(new_docs)} new documents chunks to the vectorstore.")
    else:
        print("No new documents to add.")

    return new_docs
