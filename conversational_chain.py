from clients import get_gemini_chat
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


def create_conversational_chain(retriever, llm=None):
    if llm is None:
        llm = get_gemini_chat()

    # --- STEP 1: Contextualize Question Prompt ---
    # This instructs the LLM how to rewrite the question based on history
    contextualize_q_system_prompt = """
        Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create the "Smart" Retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- STEP 2: Answer Question Prompt ---
    # This is the prompt for the final answer drafting
    system_prompt = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the user's question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
        \n\n
        {context}
        """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # This chain handles passing the 'context' (docs) into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # FINAL STEP: The full retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def chat(rag_chain):
    chat_history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = rag_chain.invoke(
            {
                "chat_history": chat_history,
                "input": user_input,
            }
        )
        print(f"\nAssistant: {response['answer']}")

        # --- HERE ARE YOUR SOURCES ---
        print("\nSources consulted:")
        for i, doc in enumerate(response["context"]):
            # PyPDFLoader usually puts the file path in metadata['source']
            source = doc.metadata.get("source", "Unknown Source")
            print(f"  [{i+1}] {source}")
        print("-" * 30)
        chat_history.extend(
            [
                HumanMessage(content=user_input),
                AIMessage(content=response["answer"]),
            ]
        )
