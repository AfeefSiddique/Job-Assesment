
# 🧠 Multilingual RAG System (English + বাংলা)

This is a simple **Multilingual Retrieval-Augmented Generation (RAG)** system that can answer user queries in **English or Bangla** based on uploaded PDF documents (e.g., **HSC Bangla 1st Paper**). It uses **Google Gemini**, **LangChain**, and **FAISS** to retrieve relevant content and generate grounded answers.

---

## 🚀 Features

✅ Accepts queries in **English and Bangla**  
✅ Uses a vector database (**FAISS**) for similarity search  
✅ Powered by **Google Gemini LLM** and **text-embedding-004**  
✅ **PDF upload + question answering** via REST API  
✅ Designed for short and accurate answers  
✅ Includes FastAPI Swagger docs at `/docs`  

---

## 📁 Folder Structure

```
📦 project_root/
├── main.py              # Main FastAPI app
├── requirements.txt     # Required libraries
├── .env                 # Google API key
└── README.md            # This file
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/multilingual-rag.git
   cd multilingual-rag
   ```

2. **Create a `.env` file** and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

5. Open the browser at [http://localhost:8000/docs](http://localhost:8000/docs) to interact with the API.

---

## 🧪 Example Queries

Uploaded PDF: `HSC26_Bangla_1st_Paper.pdf`

**Sample Bangla Questions:**
- অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? → **Expected**: শম্ভুনাথ  
- কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? → **Expected**: মামা  
- বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? → **Expected**: ১৫ বছর  

---

## 📌 API Endpoints

### 📤 `POST /upload`
Upload a PDF file. Returns a `session_id`.

- **Form Data**:
  - `file`: PDF file (e.g., HSC Bangla 1st Paper)

- **Response**:
  ```json
  {
    "session_id": "tempfile123abc.pdf",
    "message": "PDF uploaded and processed successfully."
  }
  ```

---

### ❓ `POST /ask`
Ask a question based on the uploaded document.

- **JSON Body**:
  ```json
  {
    "session_id": "tempfile123abc.pdf",
    "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
  }
  ```

- **Response**:
  ```json
  {
    "answer": "শম্ভুনাথ"
  }
  ```

---

## 🧠 Technical Overview

- **Text Extraction**:
  - Tool: `PyMuPDFLoader` from `langchain_community`
  - Reason: Handles Bangla text well with consistent structure
  - Challenge: Formatting and line breaks required cleaning

- **Chunking Strategy**:
  - Method: `RecursiveCharacterTextSplitter`
  - Chunk Size: 1000 characters
  - Why: Paragraph-level chunking preserves meaning and works well with embedding similarity

- **Embedding Model**:
  - `text-embedding-004` from Google
  - Why: Fast, multilingual, and well-supported in LangChain

- **Vector Store**:
  - Tool: FAISS (in-memory)
  - Comparison: Cosine similarity to match query vector with top document chunks

- **LLM**:
  - Model: `gemini-2.0-flash` (Google Generative AI)
  - Prompt Style: Context-prompted with instruction for concise grounded answer

---

## 📊 Evaluation (Optional Add-on)

- ✅ Groundedness: Answers come from top-matched chunks only
- ✅ Relevance: Based on FAISS cosine similarity and chunk overlap
- ❗ Can be further improved with:
  - Better chunk overlap tuning
  - Custom reranker
  - Metadata filtering

---

## 📦 requirements.txt

```
fastapi
uvicorn
python-dotenv
langchain
langchain-community
langchain-core
langchain-google-genai
langchain-text-splitters
google-generativeai
python-multipart
pydantic
PyMuPDF
```

---

## 📋 Assessment Questions & Answers

Below are the responses to the mandatory assessment questions.

---

### 1. 🧾 What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

- I used `PyMuPDFLoader` from `langchain_community.document_loaders`, which wraps around the `fitz` (PyMuPDF) library.
- **Why**: It works well for extracting structured content from PDFs and handles Bangla script with good fidelity.
- **Challenges**: Minor formatting inconsistencies like line breaks and heading overlaps, which were handled through chunking and preprocessing.

---

### 2. 📚 What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

- I used `RecursiveCharacterTextSplitter` with a **chunk size of 1000 characters**.
- This strategy ensures complete ideas are preserved in chunks and avoids cutting off mid-sentence.
- It's effective for semantic retrieval because it balances **context length** and **embedding clarity**.

---

### 3. 🧠 What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

- I used Google’s `text-embedding-004` via `langchain_google_genai`.
- **Why**: It supports **multilingual embeddings** including Bangla and provides high performance with LangChain compatibility.
- It generates dense vector representations that capture **semantic similarity** between text inputs, crucial for multilingual document search.

---

### 4. 🧮 How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

- I use **cosine similarity** with **FAISS** as the vector store.
- Cosine similarity is effective for comparing dense vector representations of text.
- FAISS is a high-performance, lightweight vector search engine ideal for **local and fast retrieval**.

---

### 5. 🧩 How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

- Query and document chunk embeddings are generated using the same embedding model to ensure they exist in the **same semantic space**.
- If the query is vague or lacks context, the system may retrieve unrelated chunks. However, the prompt instructs the LLM to say "I don't know" when insufficient context exists.

---

### 6. 🎯 Do the results seem relevant? If not, what might improve them (e.g., better chunking, better embedding model, larger document)?

- Yes, most results are relevant based on the **retrieved context + answer**.
- Potential improvements:
  - Finer-grained chunking (sentence + paragraph hybrid)
  - Custom reranker after retrieval
  - Expanded document set for broader context

---
