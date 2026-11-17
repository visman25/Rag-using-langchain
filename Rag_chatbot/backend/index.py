#load split embedding and store 
import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
load_dotenv()

# Get the folder containing main.py
BASE_DIR = Path(__file__).resolve().parent

# Go from backend/ -> Rag_chatbot/
PROJECT_DIR = BASE_DIR.parent

Data_path = PROJECT_DIR / "knowledge_base"
faiss_path = PROJECT_DIR / "faiss_index"


print("loading text files")
#Document Loader
text_loader=DirectoryLoader(Data_path,glob="**/*.txt",loader_cls=TextLoader, loader_kwargs={"encoding":"utf-8"})
txt_doc=text_loader.load()
pdf_loader=DirectoryLoader(Data_path,glob="**/*.pdf", loader_cls=PyPDFLoader)
pdf_docs=pdf_loader.load()
docs=txt_doc+pdf_docs

#Text splitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs_split=text_splitter.split_documents(docs)
print("Creating embedding")

#Embedding 
embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db= FAISS.from_documents(docs_split,embeddings)
db.save_local(faiss_path)
print("FAISS index created Successfully")

