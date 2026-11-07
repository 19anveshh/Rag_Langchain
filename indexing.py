from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from filereader import read_multiple_pdfs
from pathlib import Path
import os


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": os.getenv("HF_TOKEN")},
)

vector_store=InMemoryVectorStore(embedding=embeddings)

DOC_PATHS = [
    "pdfs/ACS QB 6.pdf",
    "pdfs/FINAL DE-QUESTION BANK.pdf",
]

docs = read_multiple_pdfs(DOC_PATHS)

text_splitter=RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap=200)

all_splits=text_splitter.split_documents(docs)

# print(f"split dpcumentation into {len(all_splits)} chunks")

vector_store.add_documents(documents=all_splits)

# print(f"indexed {len(all_splits)} chunks")