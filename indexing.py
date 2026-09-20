from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_embeddings import OpenAIEmbeddings
from langchain_chroma import Chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")  # Get the OpenAI API key from environment variables

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

file_path = DOCUMENTS_DIR / "Destiny_Lore_Summary.pdf"  # Replace with your document path



def load_document(file_path: str):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    
    documents = loader.load()
    return documents

docs = load_document(str(file_path))
print(f"Loaded {len(docs)} document(s)")
print(f"First page preview: {docs[0].page_content[:200]}...")  # Print the first 200 characters of the first page




def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    
    chunks = text_splitter.split_documents(documents)
    return chunks


document_chunks = chunk_documents(docs)
print(f"Created {len(document_chunks)} chunk(s)")
print(f"First chunk preview: {document_chunks[0].page_content[:200]}...")  # Print the first 200 characters of the first chunk




embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # You can choose a different model if needed
    api_key=api_key  # Use the API key from environment variables
    )  

sample_text = document_chunks[0].page_content  # Use the first chunk of the document
sample_embedding = embeddings.embed_query(sample_text)

print(f"Embedding dimensions: {len(sample_embedding)}")
print(f"First 10 dimensions of the embedding: {sample_embedding[:10]}")




