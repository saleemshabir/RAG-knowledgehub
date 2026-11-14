<div align="center">

# 📚 Knowledge Hub

### PDF-Based RAG Application

*Building a foundation for intelligent document understanding.*

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)
![PyPDF](https://img.shields.io/badge/PyPDF-E34F26?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)
![Swagger](https://img.shields.io/badge/Swagger-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)
![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=for-the-badge)

</div>

---

## ✨ Overview

**Knowledge Hub** is a Python-based document processing project and the foundation of a **Retrieval-Augmented Generation (RAG)** application.

Upload a PDF, extract its text, and store it locally, ready for the next stages: chunking, embeddings, semantic search, and AI-powered question answering over your own documents.

> 🚧 The project is being built step by step, with the goal of retrieving relevant information from uploaded documents and giving **context-aware answers**.

---

## 🚀 Features

| | Feature |
|---|---|
| 📤 | Upload PDF documents through a REST API |
| 📄 | Extract text from PDFs using **PyPDF** |
| 💾 | Save uploaded documents locally |
| 🗂️ | Store extracted text for further processing |
| ⚡ | Fast **FastAPI** backend with interactive docs |
| 🛡️ | Basic file validation and error handling |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python |
| **Backend** | FastAPI |
| **PDF Processing** | PyPDF |
| **Server** | Uvicorn |
| **File Handling** | Python `pathlib` |
| **API Testing** | Swagger UI |

---

## 📁 Project Structure

```
Knowledge-Hub/
│
├── app/
│   ├── app.py
│   └── main.py
│
├── data/                # Local uploads & extracted text (not tracked by Git)
│
├── requirements.txt
├── .gitignore
└── README.md
```

> 🔒 **Note:** The `data/` folder stays local and is excluded from GitHub to protect uploaded documents.

---

## ⚙️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/Knowledge-Hub-RAG.git
cd Knowledge-Hub-RAG
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate it**

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the application** (from the project root)

```bash
uvicorn app.app:app --reload
```

---

## 📖 API Documentation

Once the server is running, open the interactive Swagger UI:

👉 **http://127.0.0.1:8000/docs**

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/health` | Check application status |
| `POST` | `/upload` | Upload a PDF and extract its text |

---

## 🗺️ Development Roadmap

**Completed**

- [x] FastAPI backend setup
- [x] PDF upload functionality
- [x] PDF text extraction
- [x] Local text storage
- [x] GitHub repository setup

**Up next**

- [ ] Text chunking
- [ ] Embedding generation
- [ ] Vector database integration
- [ ] Semantic search
- [ ] Context-aware question answering

```
PDF  ──▶  Text  ──▶  Chunks  ──▶  Embeddings  ──▶  Vector DB  ──▶  Answers
 ✅        ✅          ⏳            ⏳               ⏳             ⏳
```

---

## 🎯 Future Goals

The next stages will turn Knowledge Hub into a complete RAG application by adding document chunking, vector embeddings, semantic retrieval, and an AI-powered question-answering system.

---

<div align="center">

**Knowledge Hub** – *Building a foundation for intelligent document understanding.*

⭐ If you like this project, consider giving it a star!

</div>
