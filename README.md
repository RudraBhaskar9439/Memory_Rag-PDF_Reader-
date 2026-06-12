# Memory RAG — Chat With a PDF

A Python RAG (Retrieval-Augmented Generation) tool that lets you ask questions about
a PDF in plain English. It extracts the text, embeds it locally with
`sentence-transformers`, retrieves the most relevant chunks for each question via
cosine similarity, and answers with Google's Gemini — keeping a short **conversation
memory** so follow-up questions stay coherent.

## Features
- 📄 PDF text extraction (PyMuPDF)
- ✂️ Smart chunking with overlap (LangChain `RecursiveCharacterTextSplitter`)
- 🧠 Local semantic embeddings (`all-MiniLM-L6-v2`) — no embedding API needed
- 🔎 Top-k cosine-similarity retrieval
- 💬 Conversation memory for context-aware follow-ups
- 🤖 Answer generation with Gemini

## Prerequisites
- Python 3.8+
- A Google Gemini API key

## Installation
```bash
git clone https://github.com/RudraBhaskar9439/Memory_Rag-PDF_Reader-.git
cd Memory_Rag-PDF_Reader-
pip install -r requirements.txt
```

## Usage
The API key is read from an environment variable — **never hardcode it.**
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
python MemoryRag.py path/to/your-document.pdf
```
Then ask questions:
```
Enter your question (or 'quit' to exit): What are the main topics in this document?
```

## Configuration
Tunable constants at the top of `MemoryRag.py`:
```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL   = "gemini-2.0-flash"
TOP_K       = 3      # chunks retrieved per question
```
Chunk size / overlap are arguments to `split_text_into_chunks` (default 1000 / 200),
and conversation memory length is `ConversationMemory(max_history=5)`.

## Security
⚠️ The API key is loaded from `GOOGLE_API_KEY` only. Never commit `.env` files or keys.
`.gitignore` excludes `.env`, `*.pdf`, and `__pycache__/`.

## License
MIT © 2024 Rudra Bhaskar
