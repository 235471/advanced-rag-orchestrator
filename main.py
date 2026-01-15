from load import load_data
from retrieval_strategy import hybrid_retriever
from conversational_chain import create_conversational_chain, chat
from evaluation import evaluate_rag_with_ragas
from clients import get_groq_chat, get_perplexity_chat


def main():
    # 1. Setup
    vectorstore, chunks = load_data()
    retriever = hybrid_retriever(vectorstore, chunks)
    chain = create_conversational_chain(retriever)

    # 2. Chat
    # chat(chain)

    # 4. Evaluation
    # Use Groq for Generation (fast, reliable system prompts)
    # Use Perplexity for Judging (fast, good limits, smart model)
    pplx_llm = get_perplexity_chat()
    pplx_chain = create_conversational_chain(retriever, llm=pplx_llm)
    evaluate_rag_with_ragas(pplx_chain, llm=pplx_llm)


if __name__ == "__main__":
    main()
