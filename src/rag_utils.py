"""
src/rag_utils.py

Reusable utility functions for the RAG pipeline.
These functions are used by the Jupyter notebook to keep it concise and readable.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_txt_documents(data_dir: str) -> List[Document]:
    """
    Load all .txt files from a directory.

    Args:
        data_dir: Path to the directory containing TXT files.

    Returns:
        A list of LangChain Document objects.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from '{data_dir}'.")
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into overlapping chunks.

    Args:
        documents:     List of LangChain Document objects.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s) (chunk_size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


def build_vectorstore(
    chunks: List[Document],
    index_name: str,
    namespace: str = "default",
    embedding_model: str = "text-embedding-3-small",
) -> PineconeVectorStore:
    """
    Create OpenAI embeddings and upsert document chunks into Pinecone.

    Args:
        chunks:          List of chunked Document objects to embed and store.
        index_name:      Name of the Pinecone index (must already exist).
        namespace:       Pinecone namespace to use for the vectors.
        embedding_model: OpenAI embedding model name.

    Returns:
        A LangChain PineconeVectorStore instance backed by the specified index.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
    )
    print(f"Upserted {len(chunks)} chunk(s) to Pinecone index '{index_name}' (namespace='{namespace}').")
    return vectorstore


def load_vectorstore(
    index_name: str,
    namespace: str = "default",
    embedding_model: str = "text-embedding-3-small",
) -> PineconeVectorStore:
    """
    Connect to an existing Pinecone index without upserting new documents.

    Args:
        index_name:      Name of the existing Pinecone index.
        namespace:       Pinecone namespace to use.
        embedding_model: OpenAI embedding model name.

    Returns:
        A LangChain PineconeVectorStore instance.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    return vectorstore


def format_docs(docs: List[Document]) -> str:
    """Concatenate document page content for insertion into the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    vectorstore: PineconeVectorStore,
    top_k: int = 4,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Tuple[Any, Any]:
    """
    Build a LangChain RAG chain using LCEL (LangChain Expression Language).

    The chain:
      1. Retrieves the top-k relevant chunks from Pinecone.
      2. Formats them as context.
      3. Passes context + question to an OpenAI chat model.
      4. Returns the answer as a string.

    Args:
        vectorstore: Populated PineconeVectorStore instance.
        top_k:       Number of documents to retrieve per query.
        llm_model:   OpenAI chat model name.
        temperature: Sampling temperature for the LLM.

    Returns:
        A tuple (rag_chain, retriever) where:
          - rag_chain is an LCEL Runnable that accepts {"question": str}.
          - retriever is the underlying VectorStoreRetriever (useful for inspecting sources).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use ONLY the following retrieved context to answer the question.
If you cannot find the answer in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    )

    llm = ChatOpenAI(model=llm_model, temperature=temperature)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
