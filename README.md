# 📚 BookRDR — Offline AI Book Chatbot

An offline AI-powered book assistant that answers questions from your 
personal book library using RAG (Retrieval Augmented Generation). 
No internet required after setup.

---

## What is this project?

BookRDR lets you upload any book (PDF or EPUB) and ask questions about 
it in natural language. Unlike ChatGPT or Gemini, this runs 100% on 
your own computer — no API keys, no subscriptions, no data leaving 
your machine.

---

## How does it work?
```
PDF/EPUB → Extract Text → Chunk → Embed → ChromaDB
                                              ↓
User Question → Embed → Search ChromaDB → Top 5 Chunks
                                              ↓
                                         Mistral 7B
                                              ↓
                                       Answer + Source
```

1. Books are extracted, cleaned and split into 500-word overlapping chunks
2. Each chunk is converted into a 384-dimension vector using 
   sentence-transformers
3. Vectors are stored in ChromaDB for semantic search
4. When you ask a question, it is embedded and matched against all chunks
5. Top 5 most relevant chunks are sent to Mistral 7B via Ollama
6. Mistral reads only those chunks and generates a precise answer

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| PyMuPDF + ebooklib | PDF and EPUB text extraction |
| sentence-transformers | Local embedding model (all-MiniLM-L6-v2) |
| ChromaDB | Local vector database |
| Ollama + Mistral 7B | Offline LLM for answer generation |
| Streamlit | Web chat interface |
| NumPy | Embedding storage and manipulation |

---

## Setup

### Prerequisites
- Python 3.11
- Anaconda
- [Ollama](https://ollama.com/download) installed

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/akashdwibedidj/BookRDR.git
cd BookRDR

# 2. Create conda environment
conda create -n bookrdr python=3.11
conda activate bookrdr

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download Mistral model
ollama pull mistral
```

### Create folder structure
```bash
mkdir books cleaned_texts chunks vector_db
```

### Run the app
```bash
streamlit run src/ui.py
```

---

## Usage

1. Open `http://localhost:8501` in your browser
2. Click **Add New Books** in the sidebar
3. Upload any PDF or EPUB file
4. Wait for processing to complete
5. Ask any question about your books!

---

## Project Structure
```
BookRDR/
├── src/
│   ├── phase2_extract.py      # Extract text from PDF/EPUB
│   ├── convert_to_chunks.py   # Split text into chunks
│   ├── embeding.py            # Generate embeddings
│   ├── store_to_vector.py     # Store in ChromaDB
│   ├── add_books.py           # Pipeline orchestrator
│   ├── rag.py                 # Terminal chatbot
│   └── ui.py                  # Streamlit web interface
├── .gitignore
└── README.md
```

---

## Key Features

- 100% offline — no internet needed after setup
- Add unlimited books anytime without reprocessing old ones
- Semantic search — finds meaning not just keywords
- Cites exact source book and chunk for every answer
- Rejects questions about books not in your library

---

Built with Love by Akash
```