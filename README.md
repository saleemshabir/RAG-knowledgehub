Knowledge Hub – PDF-Based RAG Application

Knowledge Hub is a Python-based document processing project and the foundation for a Retrieval-Augmented Generation (RAG) application. It allows users to upload PDF documents, extract their text, and store the extracted content locally for further processing.

The project is being developed step by step, with the goal of building a system that can retrieve relevant information from uploaded documents and provide context-aware answers.

Features

- Upload PDF documents through a REST API.
- Extract text from PDF files using PyPDF.
- Save uploaded documents locally.
- Store extracted text for further processing.
- FastAPI backend with interactive API documentation.
- Basic file validation and error handling.

Tech Stack

- Language: Python
- Backend: FastAPI
- PDF Processing: PyPDF
- Server: Uvicorn
- File Handling: Python pathlib
- API Testing: Swagger UI

Project Structure

Knowledge-Hub/
│
├── app/
│   ├── app.py
│   └── main.py
│
├── data/
│   └── (Local uploaded files and extracted text)
│
├── requirements.txt
├── .gitignore
└── README.md

Note: The data folder is kept locally and excluded from GitHub to protect uploaded documents.

Installation and Setup

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/Knowledge-Hub-RAG.git

2. Navigate to the Project

cd Knowledge-Hub-RAG

3. Create a Virtual Environment

python -m venv venv

4. Activate the Virtual Environment

Windows:

venv\Scripts\activate

5. Install Dependencies

pip install -r requirements.txt

6. Run the Application

From the project root:

uvicorn app.app:app --reload

API Documentation

Once the server is running, open:

http://127.0.0.1:8000/docs

Available endpoints:

Method| Endpoint| Description
GET| "/health"| Check application status
POST| "/upload"| Upload a PDF and extract its text

Current Development Status

- [x] FastAPI backend setup
- [x] PDF upload functionality
- [x] PDF text extraction
- [x] Local text storage
- [x] GitHub repository setup
- [ ] Text chunking
- [ ] Embedding generation
- [ ] Vector database integration
- [ ] Semantic search
- [ ] Context-aware question answering

Future Goals

The next development stages will focus on transforming Knowledge Hub into a complete RAG application by implementing document chunking, vector embeddings, semantic retrieval, and an AI-powered question-answering system.

---

Knowledge Hub – Building a foundation for intelligent document understanding.
