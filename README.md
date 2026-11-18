<div align="center">
# 📚 Knowledge Hub
### PDF-Based Retrieval-Augmented Generation (RAG) ApplicationTurning documents into intelligent, searchable knowledge.

<br>"Python" (https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
"FastAPI" (https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
"Sentence Transformers" (https://img.shields.io/badge/Sentence--Transformers-Embeddings-orange?style=for-the-badge)
"Ollama" (https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)
"NumPy" (https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
"PyPDF" (https://img.shields.io/badge/PyPDF-PDF%20Processing-E34F26?style=for-the-badge)
"Status" (https://img.shields.io/badge/Status-Core%20Pipeline%20Complete-success?style=for-the-badge)

</div>---

✨ Overview

Knowledge Hub is a Python-based document processing and Retrieval-Augmented Generation (RAG) application that allows users to retrieve relevant information from documents and generate context-aware answers using a Large Language Model (LLM).

The project combines document extraction, text chunking, semantic embeddings, similarity-based retrieval, and local AI-powered answer generation.

Instead of relying only on an LLM's pre-trained knowledge, Knowledge Hub retrieves relevant information from supplied documents before generating an answer.

«💡 The goal: Make information stored in documents searchable, accessible, and useful through natural-language questions.»

---

🚀 Features

| Feature
📤| Upload PDF documents through a REST API
📄| Extract text from PDF files using PyPDF
💾| Store uploaded and extracted documents locally
🧩| Split documents into smaller, overlapping text chunks
🔢| Generate semantic embeddings using Sentence Transformers
🔍| Retrieve relevant chunks using cosine similarity
🤖| Generate context-aware answers using a local LLM through Ollama
📚| Display source document names and chunk IDs
🎯| Filter irrelevant results using a similarity threshold
🛡️| Handle errors and out-of-scope questions
⚡| FastAPI backend with interactive Swagger documentation
📂| Process multiple text documents from a single data folder

---

🛠️ Tech Stack

Category| Technology
Language| Python
Backend| FastAPI
PDF Processing| PyPDF
Text Embeddings| Sentence Transformers
Embedding Model| "all-MiniLM-L6-v2"
Numerical Operations| NumPy
Similarity Search| Cosine Similarity
Large Language Model| Qwen 2.5 (1.5B)
LLM Runtime| Ollama
API Server| Uvicorn
API Testing| Swagger UI
File Handling| Python "pathlib"

---

📁 Project Structure

Knowledge-Hub/
│
├── app/
│   ├── app.py
│   └── main.py
│
├── data/
│   ├── input.txt
│   ├── Saleem.txt
│   └── RAG_chunks.txt
│
├── make_chunks.py
├── make_embeddings.py
├── retrieve.py
│
├── chunks.json
├── embeddings.npy
│
├── requirements.txt
├── .gitignore
└── README.md

«🔒 Note: The "data/" folder contains local documents and extracted text. It is excluded from Git tracking to avoid uploading private or personal documents.»

«📌 The current chunking script processes ".txt" files inside "data/". PDF text extraction is supported through the document-processing API, but direct PDF ingestion into the RAG chunking pipeline is a future improvement.»

---

⚙️ Installation & Setup

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/Knowledge-Hub-RAG.git
cd Knowledge-Hub-RAG

2. Create a virtual environment

python -m venv venv

3. Activate the environment

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

5. Install and configure Ollama

Install "Ollama" (https://ollama.com/) and download the model:

ollama pull qwen2.5:1.5b

Make sure Ollama is running before using the RAG question-answering pipeline.

---

🔄 How the RAG Pipeline Works

Knowledge Hub follows a Retrieval-Augmented Generation workflow.

1. Document Extraction

PDF documents are uploaded through the FastAPI backend. PyPDF extracts their text, which is stored locally for further processing.

2. Text Chunking

Large text documents are divided into smaller, manageable pieces.

Current configuration:

Parameter| Value
Chunk size| 500 characters
Chunk overlap| 50 characters

Overlapping chunks help preserve context between consecutive pieces of text.

Each chunk also stores metadata such as its unique ID and original source filename.

3. Embedding Generation

The "all-MiniLM-L6-v2" model converts each text chunk into a numerical vector.

These embeddings represent the semantic meaning of the text and allow the system to compare questions with document content.

4. Semantic Retrieval

When a user submits a question:

- The question is converted into an embedding.
- Cosine similarity measures how closely each document chunk matches the question.
- The system ranks the results and retrieves the most relevant chunks.
- A similarity threshold filters out irrelevant results.

5. Answer Generation

The retrieved information is passed as context to the Qwen 2.5 model through Ollama.

The model generates an answer based on the retrieved document content.

6. Source Attribution

The application displays the source filenames and chunk IDs used to generate the answer, making the retrieved information easier to trace.

---

▶️ Running the Project

Step 1: Generate text chunks

python make_chunks.py

Reads all ".txt" files in the "data/" directory and generates "chunks.json".

Step 2: Generate embeddings

python make_embeddings.py

Converts the generated chunks into embeddings and saves them in "embeddings.npy".

Step 3: Run the RAG application

python retrieve.py

Enter a question when prompted:

Enter your question: What is the purpose of text chunking in a RAG system?

The application retrieves relevant chunks and generates an answer using Ollama.

---

📖 API Documentation

The FastAPI backend provides an interface for document uploading and text extraction.

Start the API server from the project root:

uvicorn app.app:app --reload

Once the server is running, open Swagger UI:

👉 http://127.0.0.1:8000/docs

Method| Endpoint| Description
"GET"| "/health"| Check application status
"POST"| "/upload"| Upload a PDF and extract its text

---

🧪 Testing & Validation

The core RAG pipeline has been tested with multiple documents and different types of questions.

Test| Expected Behaviour| Result
Multiple document processing| Process all ".txt" files| ✅ Passed
Relevant question| Retrieve matching chunks| ✅ Passed
Cross-document retrieval| Retrieve information from different source files| ✅ Passed
Source attribution| Display source filename and chunk ID| ✅ Passed
Out-of-scope question| Reject irrelevant queries| ✅ Passed
Answer generation| Generate answers using retrieved context| ✅ Passed

Example Questions

Relevant question:

Who is Saleem, and what is he currently doing?

The system retrieves information from "Saleem.txt" and generates an answer using its content.

Another relevant question:

What is the purpose of text chunking in a RAG system?

The system retrieves information from "RAG_chunks.txt".

Out-of-scope question:

What is the capital of Japan?

If no relevant chunks meet the similarity threshold, the application responds:

I don't know based on the information provided.

---

🗺️ Development Roadmap

Completed

- [x] FastAPI backend setup
- [x] PDF upload functionality
- [x] PDF text extraction
- [x] Local text storage
- [x] Multi-document text processing
- [x] Text chunking with overlap
- [x] Unique chunk IDs and source metadata
- [x] Embedding generation
- [x] Semantic similarity retrieval
- [x] Top-K relevant chunk selection
- [x] Similarity threshold filtering
- [x] Local LLM integration using Ollama
- [x] Context-aware answer generation
- [x] Source citations
- [x] Out-of-scope query handling
- [x] Basic error handling and validation

Future Improvements

- [ ] Direct PDF ingestion into the RAG pipeline
- [ ] Vector database integration
- [ ] Web-based user interface
- [ ] Conversational chat history
- [ ] Improved retrieval and ranking
- [ ] Additional document format support
- [ ] Deployment and cloud hosting

---

🎯 Future Goals

The next stage of Knowledge Hub is to transform the current backend into a complete document intelligence application.

Future development will focus on a user-friendly interface, improved document management, scalable vector storage, and more advanced retrieval techniques.

The long-term goal is to create a system where users can upload their own documents and interact with their knowledge through natural-language conversations.

---

<div align="center">Knowledge Hub

From documents to knowledge. From questions to answers.

⭐ If you find this project interesting, consider giving it a star!

</div>
