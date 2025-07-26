import os
import tempfile
import dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
  # Local in-memory vector DB
import google.generativeai as genai

# Load environment variables
dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# LLM model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=500)

# Prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use one to two sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

# Chain creator
def build_rag_chain_from_pdf(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

# FastAPI app
app = FastAPI()

# Store chain temporarily (not persistent across restarts)
uploaded_pdfs = {}

class Question(BaseModel):
    query: str
    session_id: str  # Must match the one returned during upload

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await file.read())
        temp_pdf_path = temp_pdf.name

    # Create session ID from filename (could be UUID in production)
    session_id = os.path.basename(temp_pdf_path)

    # Build RAG chain from PDF
    try:
        rag_chain = build_rag_chain_from_pdf(temp_pdf_path)
        uploaded_pdfs[session_id] = rag_chain
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return {"session_id": session_id, "message": "PDF uploaded and processed successfully."}

@app.post("/ask")
async def ask_question(q: Question):
    chain = uploaded_pdfs.get(q.session_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Session ID not found. Please upload a PDF first.")

    try:
        response = chain.invoke({"input": q.query})
        return {"answer": response["answer"]}
    except Exception as e:
        return {"error": str(e)}
    
