from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from data_eval import questions, answers
from clients import get_gemini_chat, get_gemini_embeddings


def evaluate_rag_with_ragas(qa_chain, llm=None):
    if llm is None:
        llm = get_gemini_chat()

    llm_answers = []
    retrieved_context = []

    for question in questions:
        # The chain expects "input" and "chat_history"
        response = qa_chain.invoke({"input": question, "chat_history": []})
        llm_answers.append(response["answer"])
        # In the new LCEL chain, documents are returned in the "context" key
        retrieved_context.append([doc.page_content for doc in response["context"]])

    dataset_dict = {
        "question": questions,
        "answer": llm_answers,
        "contexts": retrieved_context,
        "ground_truth": answers,
    }

    dataset = Dataset.from_dict(dataset_dict)

    eval_result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=get_gemini_embeddings(),
    )

    df_result = eval_result.to_pandas()
    df_result.to_csv("evaluation_results.csv", index=False)
    print("\nEvaluation Results:")
    print(df_result)
    return eval_result
