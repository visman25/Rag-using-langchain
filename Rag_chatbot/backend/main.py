import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional
from pathlib import Path
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Go from backend/ -> Rag_chatbot/
PROJECT_DIR = BASE_DIR.parent

faiss_path = PROJECT_DIR / "faiss_index"
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(faiss_path,embeddings,allow_dangerous_deserialization=True)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto",
    max_new_tokens=100,
)

model = ChatHuggingFace(llm=llm)
retriever=db.as_retriever(search_kwargs={"k":3})
System_prompt=''' 
You are a helpful assistant.
 Answer only from the provided transcript context.
    If context is insufficient, just say you dont know.
    Context: 000000{context}
'''
prompt=ChatPromptTemplate.from_messages([("system",System_prompt),("human","{input}")])

qa_chain=create_stuff_documents_chain(model,prompt)
rag_chain=create_retrieval_chain(retriever,qa_chain)

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

class Query(BaseModel):
    text:str
@app.get("/")
def home():
    return{"message":"rag is running"}

@app.post("/query")
def query_rag(query: Query):
    response = rag_chain.invoke({"input": query.text})

    return {
        "answer": response.get("answer", "No answer found")
    }