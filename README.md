# Biomedical RAG System for Literature Analysis

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for biomedical literature analysis. It retrieves research abstracts from PubMed, processes them into embeddings, stores them in a vector database, and uses a large language model to generate answers grounded in retrieved context.

The system is designed to assist in querying biomedical research topics such as drug repurposing, disease mechanisms, and therapeutic pathways.

---

## Architecture

The pipeline follows this structure:

```
User Query → Embedding → Vector Search (ChromaDB) → Context Retrieval → LLM (Groq) → Answer
```

### Components

* Data Source: PubMed abstracts retrieved using Biopython Entrez API
* Embedding Model: SentenceTransformers (`all-MiniLM-L6-v2`)
* Vector Database: ChromaDB (persistent storage)
* LLM: Groq (LLaMA 3.1 8B Instant)
* Language: Python

---

## Features

* Automated retrieval of biomedical abstracts from PubMed
* Text cleaning and chunking for efficient embedding
* Persistent vector storage using ChromaDB
* Semantic search using embedding similarity
* Context-aware answer generation using LLM
* Strict grounding of answers in retrieved context

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install chromadb sentence-transformers groq python-dotenv biopython
```

---

## Environment Setup

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_api_key_here
```

---

## Usage

Run the main script:

```bash
python main.py
```

Enter a query when prompted, for example:

```
Enter query: Which drugs can be repurposed for cancer?
```

---

## Workflow

1. Fetch abstracts from PubMed based on a query
2. Clean and preprocess text
3. Split text into overlapping chunks
4. Convert chunks into embeddings
5. Store embeddings in ChromaDB
6. Retrieve relevant chunks based on user query
7. Generate answer using Groq LLM

---

## Project Structure

```
.
├── main.py
├── .env
├── chroma_db/
└── README.md
```

---

## Important Notes

* Abstracts are used instead of full papers, which may limit answer completeness




