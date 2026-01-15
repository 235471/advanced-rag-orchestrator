"""
Quota-Friendly RAG Evaluation Module

This module provides a rate-limit aware evaluation function for RAG systems.
It processes one question at a time to avoid hitting API quotas.

Supports: Gemini, Groq, and Perplexity as LLM providers.
"""

import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from data_eval import questions, answers
from clients import (
    get_gemini_chat,
    get_gemini_embeddings,
    get_groq_chat,
    get_perplexity_chat,
)


# Provider mapping for easy switching
LLM_PROVIDERS = {
    "gemini": get_gemini_chat,
    "groq": get_groq_chat,
    "perplexity": get_perplexity_chat,
}


def evaluate_rag_quota_friendly(qa_chain, llm_provider="gemini", sleep_between_evals=0):
    """
    Evaluate a RAG chain with a single question for quota-constrained environments.

    Args:
        qa_chain: The conversational RAG chain to evaluate.
        llm_provider: One of "gemini", "groq", or "perplexity". Default is "gemini".
        sleep_between_evals: Seconds to sleep between evaluations. Use 60 for strict quotas.

    Returns:
        pd.DataFrame: Evaluation results.
    """
    if llm_provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Invalid llm_provider. Choose from: {list(LLM_PROVIDERS.keys())}"
        )

    llm_getter = LLM_PROVIDERS[llm_provider]
    llm = llm_getter()

    llm_answers = []
    retrieved_context = []

    print(f"\n--- Starting Generation Phase (Single Item Mode) ---")
    print(f"LLM Provider: {llm_provider.upper()}")

    # 1. Generation Phase - Limit to just the first question for quota safety
    questions_subset = questions[:1]

    for i, question in enumerate(questions_subset):
        print(f"Generating answer for question {i+1}/{len(questions_subset)}...")

        # Invoke chain
        response = qa_chain.invoke({"input": question, "chat_history": []})

        llm_answers.append(response["answer"])
        retrieved_context.append([doc.page_content for doc in response["context"]])

        # Small sleep just to be polite to the generation API
        time.sleep(2)

    # 2. Evaluation Phase (The heavy API user)
    print("\n--- Starting Evaluation Phase (Slow Mode) ---")

    # We will accumulate results in a list of dicts
    results_list = []

    # Ragas requires a dataset object, but we will minimize it to 1 row at a time
    for i in range(len(questions_subset)):
        print(f"Evaluating {i+1}/{len(questions_subset)}...")

        # Create a mini-dataset for just this one row
        single_row_dict = {
            "question": [questions_subset[i]],
            "answer": [llm_answers[i]],
            "contexts": [retrieved_context[i]],
            "ground_truth": [answers[i]],
        }
        dataset = Dataset.from_dict(single_row_dict)

        try:
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
                raise_exceptions=False,
            )

            # Extract the scores
            scores = eval_result.to_pandas().to_dict("records")[0]

            # Combine input data and scores into a flat record
            record = {
                "question": questions_subset[i],
                "answer": llm_answers[i],
                "ground_truth": answers[i],
                **scores,
            }
            results_list.append(record)
            print("  -> Success")

        except Exception as e:
            print(f"  -> Failed for question {i+1}: {e}")

        # Optional sleep to respect quota
        if sleep_between_evals > 0:
            print(f"  -> Sleeping {sleep_between_evals}s to reset quota...")
            time.sleep(sleep_between_evals)

    # 3. Create Final DataFrame
    df_result = pd.DataFrame(results_list)
    df_result.to_csv("evaluation_results_quota.csv", index=False)

    print("\nEvaluation Results (Quota Safe):")
    print(df_result)

    return df_result
