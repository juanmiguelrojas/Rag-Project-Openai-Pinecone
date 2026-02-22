# RAG Project – OpenAI + Pinecone

> **Lab:** Introduction to Creating RAGs (Retrieval-Augmented Generators) with OpenAI  
> **Repository 2** – Full implementation using LangChain v0.2+, OpenAI, and Pinecone.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Setup & Installation](#setup--installation)
6. [Environment Variables](#environment-variables)
7. [Running the Notebook](#running-the-notebook)
8. [Example Outputs](#example-outputs)
9. [Key Components](#key-components)
10. [References](#references)

---

## Project Overview

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline that:

1. Loads plain-text (`.txt`) documents from a local `data/` directory.
2. Splits them into overlapping chunks.
3. Embeds the chunks with **OpenAI `text-embedding-3-small`**.
4. Stores / retrieves vectors in **Pinecone** (serverless or pod-based index).
5. Answers natural-language questions by retrieving relevant chunks and passing them as context to **OpenAI `gpt-4o-mini`** via a **LangChain LCEL** chain.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  data/*.txt  ──►  DirectoryLoader/TextLoader                    │
│                         │                                        │
│                         ▼                                        │
│              RecursiveCharacterTextSplitter                      │
│           (chunk_size=1000, chunk_overlap=200)                   │
│                         │                                        │
│                         ▼                                        │
│             OpenAIEmbeddings (text-embedding-3-small)            │
│                         │                                        │
│                         ▼                                        │
│              PineconeVectorStore.from_documents()                │
│                    (upsert vectors)                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                           │
│                                                                  │
│  user question                                                   │
│       │                                                          │
│       ├──► VectorStoreRetriever ──► top-k chunks from Pinecone  │
│       │                                    │                     │
│       │                             format_docs()               │
│       │                                    │                     │
│       └───────────────────────────► ChatPromptTemplate          │
│                                            │                     │
│                                   ChatOpenAI (gpt-4o-mini)      │
│                                            │                     │
│                                     StrOutputParser             │
│                                            │                     │
│                                      final answer               │
└─────────────────────────────────────────────────────────────────┘
```

**Components summary:**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Document loader | LangChain `DirectoryLoader` + `TextLoader` | Load `.txt` files |
| Text splitter | `RecursiveCharacterTextSplitter` | Chunk documents |
| Embeddings | OpenAI `text-embedding-3-small` | Vectorise chunks |
| Vector store | Pinecone (via `langchain-pinecone`) | Store & retrieve vectors |
| Retriever | `VectorStoreRetriever` | Similarity search (top-k) |
| LLM | OpenAI `gpt-4o-mini` | Generate answers |
| Chain | LangChain LCEL | Wire retriever → prompt → LLM |

---

## Project Structure

```
Rag-Project-Openai-Pinecone/
├── .env.example                          # Environment variable template
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
│
├── data/
│   ├── README.md                         # Instructions for adding TXT files
│   ├── ai_overview.txt                   # Sample document – AI overview
│   ├── ml_basics.txt                     # Sample document – ML basics
│   └── llm_intro.txt                     # Sample document – LLM introduction
│
├── notebooks/
│   └── 01_rag_openai_pinecone_txt.ipynb  # Main Jupyter notebook
│
└── src/
    ├── __init__.py
    └── rag_utils.py                      # Reusable pipeline helper functions
```

---

## Prerequisites

- Python 3.9+
- An **OpenAI** account with an API key – [https://platform.openai.com](https://platform.openai.com)
- A **Pinecone** account with an API key – [https://app.pinecone.io](https://app.pinecone.io)
- A Pinecone index created with **dimension = 1536** and **metric = cosine**

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/juanmiguelrojas/Rag-Project-Openai-Pinecone.git
cd Rag-Project-Openai-Pinecone
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 5. Create the Pinecone index

Either via the [Pinecone Console](https://app.pinecone.io) or using **Step 4** in the notebook.

| Setting | Value |
|---------|-------|
| Name | value of `PINECONE_INDEX_NAME` in your `.env` |
| Dimensions | `1536` |
| Metric | `cosine` |

### 6. Launch Jupyter and open the notebook

```bash
jupyter notebook notebooks/01_rag_openai_pinecone_txt.ipynb
```

Run all cells from top to bottom.

---

## Environment Variables

Copy `.env.example` to `.env` and provide the following values:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `PINECONE_API_KEY` | ✅ | Pinecone API key |
| `PINECONE_INDEX_NAME` | ✅ | Name of your Pinecone index |
| `PINECONE_NAMESPACE` | ❌ | Namespace inside the index (default: `default`) |

`.env.example`:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-openai-index
PINECONE_NAMESPACE=default
```

---

## Running the Notebook

The notebook (`notebooks/01_rag_openai_pinecone_txt.ipynb`) is divided into 9 steps:

| Step | Description |
|------|-------------|
| 1 | Load environment variables from `.env` |
| 2 | Load TXT documents from `data/` |
| 3 | Split documents into overlapping chunks |
| 4 | Verify / create the Pinecone index |
| 5 | Embed chunks and upsert vectors to Pinecone |
| 6 | *(Optional)* Connect to an existing vectorstore (skip re-ingestion) |
| 7 | Build the LCEL RAG chain |
| 8 | Run example RAG queries |
| 9 | Inspect retrieved source documents |

---

## Example Outputs

### Loading documents (Step 2)

```
100%|██████████| 3/3 [00:00<00:00, 45.12it/s]
Loaded 3 document(s) from '../data'.

--- First document preview ---
Source : ../data/ai_overview.txt
Length : 1560 characters

First 300 characters:
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines...
```

### Splitting (Step 3)

```
Split into 8 chunk(s) (chunk_size=1000, overlap=200).

Total chunks : 8

--- Sample chunk ---
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
especially computer systems. These processes include learning ...

Metadata: {'source': '../data/ai_overview.txt'}
```

### Upsert to Pinecone (Step 5)

```
Upserted 8 chunk(s) to Pinecone index 'rag-openai-index' (namespace='default').

✅  Vectors upserted successfully.
```

### RAG query (Step 8)

```
❓ Question: What is Retrieval-Augmented Generation (RAG) and why is it useful?

💬 Answer:
Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs)
by coupling them with a retrieval system. Instead of relying solely on parametric knowledge
(the model's weights), the model retrieves relevant documents from an external knowledge base
at inference time, grounding its responses in factual, up-to-date information. RAG reduces
hallucinations and allows the model to cite its sources.
```

### Inspecting sources (Step 9)

```
🔍  Retrieved 4 chunk(s) for: "What are the limitations of Large Language Models?"

── Chunk 1 ── source: ../data/llm_intro.txt
Large Language Models (LLMs) are a type of artificial intelligence model trained on vast
amounts of text data to understand and generate human-like language ...

── Chunk 2 ── source: ../data/llm_intro.txt
Limitations of LLMs include hallucinations (generating plausible-sounding but incorrect
information), knowledge cutoffs, high computational cost ...

📚  Sources used:
 - ../data/llm_intro.txt
 - ../data/llm_intro.txt
 - ../data/ai_overview.txt
 - ../data/ml_basics.txt
```

---

## Key Components

### `src/rag_utils.py`

Reusable helper functions to keep the notebook readable:

| Function | Description |
|----------|-------------|
| `load_txt_documents(data_dir)` | Loads all `.txt` files from a directory |
| `split_documents(docs, chunk_size, chunk_overlap)` | Splits documents into overlapping chunks |
| `build_vectorstore(chunks, index_name, namespace, embedding_model)` | Embeds and upserts chunks to Pinecone |
| `load_vectorstore(index_name, namespace, embedding_model)` | Connects to an existing Pinecone index |
| `build_rag_chain(vectorstore, top_k, llm_model, temperature)` | Builds the LCEL RAG chain |

---

## References

- [LangChain LLM Chain Quickstart](https://python.langchain.com/docs/tutorials/llm_chain/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Pinecone Integration](https://python.langchain.com/docs/integrations/vectorstores/pinecone/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
