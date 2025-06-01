
# sentences = ["Rudra Mota", "Rudra Bad", "Meow", "Bhow"]

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)

# # print(embeddings)
# print(type(embeddings))
# print(len(embeddings))
# print(len(embeddings[0]))
# print(embeddings[0])

# # 2d Array : List of List of Numbers
# # 4
# # 384
# # list of 384 Number

from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import google.generativeai as genai
import os
from typing import List, Dict


class ConversationMemory:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add_interaction(self, query: str, response: str, context: str):
        self.history.append({
            "query": query,
            "response": response,
            "context": context
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_formatted_history(self) -> str:
        formatted = ""
        for interaction in self.history:
            formatted += f"Question: {interaction['query']}\n"
            formatted += f"Answer: {interaction['response']}\n"
        return formatted

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text += page.get_text()
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)
    # chunks = []
    # start = 0
    # text_length = len(text)
    # while start < text_length:
    #     end = min(start + chunk_size, text_length)
    #     chunks.append(text[start:end])
    #     start += chunk_size - chunk_overlap
    # return chunks


def sentence_encode(sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return embeddings

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
if __name__ == "__main__":
    pdf_path = "RudraCV.pdf"  
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
  
    # print(type(chunks)) # List of st
    # print(type(chunks[0])) # str
    # print(type(chunks[0][0])) # char
    # print(len(chunks)) # Total Charecter / 1000 = 14
    # print(len(chunks[0]))  # 1000

    chunk_vectors = []
    chunk_vectors = sentence_encode(chunks)
    
#############################
        # Initialize conversation memory
    memory = ConversationMemory()
#############################


    # print(type(chunk_vectors))  # List of List of Numbers
    # print(type(chunk_vectors[0])) # List of Numbers
    # print(len(chunk_vectors))  # 14
    # print(len(chunk_vectors[0])) # 384

    # query = "Tell me about Hackathons rudra has participated in ?"
    # query_vector = sentence_encode([query])
    # print(len(query_vector))  # 1
    # print(len(query_vector[0]))  # 384
    while True:
        # Get user input
        query = input("\nEnter your question (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break
            
        query_vector = sentence_encode([query])
        top_k = 3
        
        similarities = []
        for idx, chunk_vec in enumerate(chunk_vectors):
            sim = cosine_similarity(chunk_vec, query_vector[0])
            similarities.append((sim, idx))
        
        print("Similarities:", similarities)

        print("==" * 20)

        # Sort by similarity descending and get top_k indices
        top_chunks = sorted(similarities, reverse=True)[:top_k]
        top_indices = [idx for _, idx in top_chunks]

        print("Top chunk indices:", top_indices)

        new_context = ""
        for i in top_indices:
            new_context += chunks[i] + "\n"

    #     prompt_template = f"""You are a helpful assistant. Answer the question based on the context provided.
    # Context: {new_context}
    # Question: {query}"""

        GOOGLE_API_KEY = "AIzaSyBqqT1xFUv4iMDViJ8dIJHlD_kM7_T0fE4"

    # client = genai.Client(api_key=GOOGLE_API_KEY)

    # response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents="prompt_template",
    # )

    # print(response.text)
        

        # prompt_template = f"""You are a helpful assistant. Answer the question based on the context provided.
        # Context: {new_context}
        # Question: {query}"""
###########################Updating the prompt template
          # Create history-aware prompt
        conversation_history = memory.get_formatted_history()
        prompt_template = f"""You are a helpful assistant with access to previous conversation context and the current question.

Previous Conversation:
{conversation_history}

Current Context:
{new_context}

Current Question: {query}

Please provide a coherent answer that takes into account both the conversation history and the current context."""
        try:
                # Configure the API
                genai.configure(api_key=GOOGLE_API_KEY)
                
                # Initialize the model correctly
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Generate response with the actual prompt
                response = model.generate_content(prompt_template)
                print("\nResponse:")
                print(response.text)
                # Store interaction in memory
                memory.add_interaction(query, response.text, new_context)
        except Exception as e:
                print(f"Error generating response: {str(e)}")