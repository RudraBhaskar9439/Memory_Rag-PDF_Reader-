# PDF Knowledge Base with Gemini AI
A Python application that creates an intelligent knowledge base from multiple PDF documents using semantic search and Google's Gemini AI, with conversation memory and local storage capabilities.

# Features
📚 Multi-PDF document processing
🧠 Semantic search using sentence transformers
💾 Local storage of processed data
🤖 Integration with Google's Gemini AI
💬 Conversation memory for context-aware responses
🔄 Automatic detection of PDF updates
Prerequisites
Python 3.8+
Google Gemini API key
PDF documents to process

# Installation
Clone the repository:
```python
git clone <your-repo-url>
cd <your-repo-directory>
```
Install required packages:
```python
pip3 install google-generativeai sentence-transformers PyMuPDF langchain python-dotenv numpy
```
Create a .env file in the project root:
```python
GOOGLE_API_KEY=your_gemini_api_key_here
```
Project Structure
```python
.
├── LocallyStored.py        # Main application file
├── Folder_with_pdfs/       # Directory for PDF files
├── processed_data.pkl      # Cached processed data
├── .env                    # Environment variables
└── README.md              # This file
```
Usage
Place your PDF files in the Folder_with_pdfs directory
Run the script:
```python
python3 LocallyStored.py
```


Interact with your documents:
```python
Enter your question (or 'quit' to exit): What are the main topics covered in the PDFs?
```
# Features in Detail
 ## PDF Processing
1. Automatic text extraction from PDFs 

2. Smart text chunking with overlap

3. Source tracking for each chunk

4. Embedding generation using SentenceTransformer

## Semantic Search

1. Cosine similarity for relevance matching

2. Top-k chunk retrieval

3. Source-aware context building

## Conversation Memory

1. Maintains conversation history

2. Context-aware responses

3. Configurable memory size

## Local Storage

1. Caches processed documents

2. Saves conversation history
3. Detects document updates

4. Reduces processing time

# Configuration
You can customize these parameters in the code:
```python
chunk_size = 1000          # Size of text chunks
chunk_overlap = 200        # Overlap between chunks
max_history = 5           # Number of conversations to remember
top_k = 3                # Number of relevant chunks to retrieve
```
# Error Handling
The application includes error handling for:

Missing API keys

PDF processing errors

File system issues

Model generation failures

Empty responses

# Contributing

Fork the repository

Create your feature branch

Commit your changes

Push to the branch


Create a Pull Request

# MIT License

Copyright (c) 2024 [Rudra Bhaskar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Security Note
⚠️ Never commit your .env file or expose your API keys.

Author
[Rudra Bhaskar]
