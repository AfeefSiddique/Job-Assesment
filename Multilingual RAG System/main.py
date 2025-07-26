import os
import dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import google.generativeai as genai

# Load environment variables
dotenv.load_dotenv()
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load PDF
loader = PyMuPDFLoader(r"F:\BrainStation 23\LLM-Langchain\Books\HSC26-Bangla1st-Paper.pdf")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Vectorstore
index_name = "harrypotter-qna"  # Make sure this Pinecone index exists
vectorstore = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

# Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 50})

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=500)

# Prompt Template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use one to two sentences maximum and keep the "
    "answer concise. "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# FastAPI app
app = FastAPI()

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(q: Question):
    try:
        response = rag_chain.invoke({"input": q.query})
        return {"answer": response["answer"]}
    except Exception as e:
        return {"error": str(e)}
