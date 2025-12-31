import os 
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all documents from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # Check if directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please add it to company files")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,  # Path of the document
        glob="*.txt",  # only loads documents with .txt. Can be modified (.pdf,.png,.mp4) later
        loader_cls=TextLoader,  # our loader class since we are working with text data
        loader_kwargs={"encoding": "utf-8"},  # ensure files are read as UTF-8 to avoid decode errors
    )

    documents=loader.load() # Loads all the documents 
    
    if len(documents)==0:
        raise FileNotFoundError(f"No .txt files found at {docs_path}. Please add company documents")
    
    for i,doc in enumerate(documents[:2]):# Shows 1st 2 documents
        print(f"\n Document {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Doc lenght: {len(doc.page_content)} characters")
        print(f"  Content preview : {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents,chunk_size=800,chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chenks")

    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks=text_splitter.split_documents(documents)# Splits documents into chunks

    if chunks:
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)
        
        if len(chunks)>5:
            print(f"\n... and {len(chunks)-5} more chunks")
    return chunks

def create_vector_store(chunks,persist_directory='db/chroma_db'):
    """Create and persist ChromaDB vector store"""

    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")

    #Create ChromaDB Vector store
    print("--Creating Vector Store--")
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}# Define the algorith for similarity, in this case cosine similarity
    )

    print("--Finished creating Vector Store--")
    print(f"vector Store created and saved to {persist_directory}")
    return vectorstore

def main():
    """Main Ingestion Pipeline"""

    docs_path="docs"
    persistent_directory="db/chroma_db"

    #Checking if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Persistent directory already exists. No need to re-process documents.")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore

    print("Persistent directory does not exist. Initializing vector store...\n")
    
    #Step 1 : Load the Documents
    documents=load_documents(docs_path)

    #Step 2 : Chunking
    chunks=split_documents(documents)

    #Step 3 : Create Vector Store
    vectorstore=create_vector_store(chunks,persistent_directory)

    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__=="__main__":
    main()












# documents list  is going to loook like 
    # documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]