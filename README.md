# 🧠 LangChain RAG Capstone Project

A **production-ready Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, featuring hybrid search, conversational memory, and comprehensive evaluation with RAGAS.

## 🌟 Features

- **Hybrid Search**: Combines BM25 (lexical) + Semantic (vector) retrieval for robust document retrieval
- **Conversational Memory**: Context-aware chat with history using LCEL chains
- **Multi-Provider Support**: Gemini, Groq, and Perplexity integration
- **RAGAS Evaluation**: Automated quality metrics (Faithfulness, Answer Relevancy, Context Precision/Recall)
- **Quota-Friendly Mode**: Rate-limit aware evaluation for free-tier APIs
- **Deduplication**: Source-aware document hashing to prevent duplicate chunks

## 📁 Project Structure

```
langchain-capstone/
├── main.py                    # Main orchestration script
├── clients.py                 # API clients (Gemini, Groq, Perplexity, Chroma)
├── load.py                    # Document loading and vectorstore management
├── ingest.py                  # PDF document ingestion
├── transform.py               # Text chunking with RecursiveCharacterTextSplitter
├── retrieval_strategy.py      # Hybrid retriever (BM25 + Semantic)
├── conversational_chain.py    # LCEL-based conversational RAG chain
├── evaluation.py              # Full RAGAS evaluation
├── evaluation_quota.py        # Quota-friendly single-question evaluation
├── data_eval.py               # Ground truth Q&A pairs for evaluation
├── knowledge_base/            # PDF documents for RAG
└── db/                        # ChromaDB persistence directory
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
PPLX_API_KEY=your_perplexity_api_key
```

### 3. Add Documents

Place your PDF files in the `knowledge_base/` directory.

### 4. Run the Pipeline

```bash
python main.py
```

## 🔧 Configuration

### Retrieval Settings (`retrieval_strategy.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 10 | Number of chunks to retrieve |
| `weights` | [0.38, 0.62] | BM25 vs Semantic weight ratio |

### Chunking Settings (`transform.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |

## 📊 Evaluation

### Full Evaluation (High Quota)

```python
from evaluation import evaluate_rag_with_ragas

evaluate_rag_with_ragas(chain, llm=pplx_llm)
```

### Quota-Friendly Evaluation (Low Quota)

```python
from evaluation_quota import evaluate_rag_quota_friendly

# Options: "gemini", "groq", "perplexity"
evaluate_rag_quota_friendly(chain, llm_provider="perplexity", sleep_between_evals=60)
```

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Is the answer supported by the retrieved context? |
| **Answer Relevancy** | Does the answer address the user's question? |
| **Context Precision** | Are relevant documents ranked higher? |
| **Context Recall** | Were all necessary documents retrieved? |

## 🤖 Supported LLM Providers

| Provider | Model | Use Case |
|----------|-------|----------|
| **Gemini** | gemini-3-flash-preview | Default, good for general use |
| **Groq** | llama-3.3-70b-versatile | Fast inference, system prompt support |
| **Perplexity** | sonar | High rate limits, reliable for evaluation |

## 📈 Sample Results

```
| Question                                  | Faithfulness | Answer Relevancy | Context Recall |
|-------------------------------------------|--------------|------------------|----------------|
| O que é RAG e qual problema ele soluciona?| 0.94         | 0.77             | 1.0            |
| Quais os componentes essenciais do RAG?   | 1.0          | 0.80             | 1.0            |
| Qual a diferença entre busca lexical...   | 1.0          | 0.84             | 1.0            |
| O que mede a métrica faithfulness...      | 0.33         | 0.87             | 1.0            |
```

## 🛠️ Development

### Reset Vector Database

```bash
rm -rf ./db
python main.py  # Will re-index documents
```

### Interactive Chat Mode

```python
# In main.py, uncomment:
chat(chain)
```

## 📚 Technologies Used

- **LangChain** - RAG framework
- **ChromaDB** - Vector database
- **RAGAS** - Evaluation metrics
- **Gemini/Groq/Perplexity** - LLM providers
- **BM25** - Lexical search
- **HuggingFace Datasets** - Evaluation data handling

## 🙏 Acknowledgments

Built as part of the LangChain certification program.

# License

Free to use and modify.