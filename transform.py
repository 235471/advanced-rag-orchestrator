from langchain_text_splitters import RecursiveCharacterTextSplitter


def transform_documents_chunksize(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunk = text_splitter.split_documents(documents)
    return chunk
