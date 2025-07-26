import os
import tempfile
import dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import pinecone
import google.generativeai as genai

# Load environment variables
dotenv.load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # e.g., "us-west1-gcp"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

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

# FastAPI app
app = FastAPI()

# Store chains per session (replace with DB in production)
uploaded_pdfs = {}

# Replace this with your Pinecone index name
INDEX_NAME = "rag-bot"

# Make sure index exists or create it (adjust dimension according to embedding size)
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=1536)  # Adjust dimension if needed

class Question(BaseModel):
    query: str
    session_id: str  # session to identify which index or vectorstore

def build_rag_chain_from_pdf(pdf_path: str, session_id: str):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Create Pinecone vectorstore scoped by index_name
    vectorstore = PineconeVectorStore.from_documents(
        docs,
        embeddings,
        index_name=INDEX_NAME,
        namespace=session_id  # use session_id as namespace to isolate user data
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await file.read())
        temp_pdf_path = temp_pdf.name

    session_id = os.path.basename(temp_pdf_path)

    try:
        rag_chain = build_rag_chain_from_pdf(temp_pdf_path, session_id)
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
