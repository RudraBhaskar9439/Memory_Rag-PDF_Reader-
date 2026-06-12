"""
Memory RAG — chat with a PDF, with conversation memory.

Pipeline: extract text from a PDF -> chunk it -> embed chunks locally with
sentence-transformers -> on each question, retrieve the top-k most similar
chunks (cosine) and answer with Gemini, keeping a short conversation history
so follow-up questions stay coherent.

Usage:
    export GOOGLE_API_KEY="your-key"        # never hardcode it
    python MemoryRag.py path/to/document.pdf
"""

import os
import sys
from typing import Dict, List

import fitz  # PyMuPDF
import numpy as np
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "gemini-2.0-flash"
TOP_K = 3

# Load the embedding model once at import (it's expensive to construct).
_embedder = SentenceTransformer(EMBED_MODEL)


class ConversationMemory:
    """Keeps the last `max_history` Q/A turns so follow-ups have context."""

    def __init__(self, max_history: int = 5):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add_interaction(self, query: str, response: str, context: str):
        self.history.append({"query": query, "response": response, "context": context})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_formatted_history(self) -> str:
        return "".join(
            f"Question: {turn['query']}\nAnswer: {turn['response']}\n"
            for turn in self.history
        )


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_text(text)


def encode(sentences: List[str]) -> np.ndarray:
    return _embedder.encode(sentences)


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GOOGLE_API_KEY in your environment first (do NOT hardcode it).")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEN_MODEL)

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdf_path or not os.path.exists(pdf_path):
        sys.exit("Usage: python MemoryRag.py path/to/document.pdf")

    print(f"Reading {pdf_path} ...")
    chunks = split_text_into_chunks(extract_text_from_pdf(pdf_path))
    chunk_vectors = encode(chunks)
    print(f"Indexed {len(chunks)} chunks. Ask away.\n")

    memory = ConversationMemory()
    while True:
        query = input("Enter your question (or 'quit' to exit): ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        query_vector = encode([query])[0]
        scored = [(cosine_similarity(vec, query_vector), idx) for idx, vec in enumerate(chunk_vectors)]
        top_indices = [idx for _, idx in sorted(scored, reverse=True)[:TOP_K]]
        context = "\n".join(chunks[i] for i in top_indices)

        prompt = (
            "You are a helpful assistant. Answer using the conversation history and the "
            "current context.\n\n"
            f"Previous Conversation:\n{memory.get_formatted_history()}\n"
            f"Current Context:\n{context}\n\n"
            f"Current Question: {query}"
        )
        try:
            response = model.generate_content(prompt)
            print(f"\nResponse:\n{response.text}\n")
            memory.add_interaction(query, response.text, context)
        except Exception as e:
            print(f"Error generating response: {e}")


if __name__ == "__main__":
    main()
