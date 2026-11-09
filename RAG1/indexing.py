from dotenv import load_dotenv
from file_reader import extract_text

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Vector store
vector_store = InMemoryVectorStore(embedding)

# Extract text
raw_text = extract_text("pdfs/Penguins_ACL.pdf")

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(raw_text)

print(f"Created {len(chunks)} chunks")

# Convert chunks to documents
documents = [
    Document(page_content=chunk)
    for chunk in chunks
]

# Store embeddings
vector_store.add_documents(documents)

print("Indexing complete")