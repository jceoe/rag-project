from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
from pathlib import Path



def run_rag_pipeline(file_path: str):

    # ==========================
    # Load Documents
    print(f"\nLoading document from: {file_path}")
    
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load() # Becomes a langchain document object with page_content and metadata attributes
    
    print(f"Loaded {len(documents)} document(s)")
    # ==========================
    
    

    # ==========================
    # Chunk documents
    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
        
    document_chunks = text_splitter.split_documents(documents)

    print(f"Created {len(document_chunks)} chunk(s)")
    # ==========================
    
    

    # ==========================
    # Initialise OpenAI Embeddings
    print("\nGenerating embeddings for document chunks...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # You can choose a different model if needed
        api_key=os.getenv("OPENAI_API_KEY")  # Use the API key from environment variables
    )  

    print("Embeddings generated successfully.")
    # ==========================
    
    

    # ==========================
    # Create vector store
    print("\nCreating vector store with Chroma...")

    vectorstore = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Directory to persist the vector store
    )

    print(f"Vector store created and persisted at './chroma_db' with {len(document_chunks)} chunk(s).")


    # test
    
    #query = "Using only the information in the document, explain what the Collapse is and how it happened." # Replace with your query in relation to the document content
    #results = vectorstore.similarity_search(query, k=3)  # Retrieve top 3 similar chunks

    #print("\nTesting similarity search...")
    #print(f" Query: '{query}'")
    #print(f"Found {len(results)} similar chunks:")
    
    #print(f"Top 3 similar chunks for query '{query}':")
    #for i, result in enumerate(results, start=1):
    #    print(f"{i}. {result.page_content[:200]}...")

    return vectorstore  # Return the vector store for further use (e.g., retrieval, generation)
    
# ==========================
# Main execution block

if __name__ == "__main__":
    
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")  # Get the OpenAI API key from environment variables

    BASE_DIR = Path(__file__).resolve().parent
    DOCUMENTS_DIR = BASE_DIR / "documents"
    file_path = DOCUMENTS_DIR / "Destiny_Lore_Summary.pdf"  # Replace with your document path

    vectorstore = run_rag_pipeline(str(file_path))