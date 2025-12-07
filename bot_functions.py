import os
# Import your specific LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv
 

# --- Configuration & Initialization ---
load_dotenv()

# Define global objects/initialize connection outside the function
PINECONE_INDEX_NAME = "langchain-gen-search"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINIAI_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GEMINI_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", 
    google_api_key=GEMINI_API_KEY,
    output_dimensionality=3072
)

pc = Pinecone(api_key=PINECONE_API_KEY)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

# --- Your RAG Function ---

def run_qa(query: str, vectorstore, llm, k: int = 3) -> str:
    # Function body remains the same
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based ONLY on the following context:\n{context}\n\nQuestion:\n{query}"
    response = llm.invoke(prompt)
    answer = response.content
    return answer