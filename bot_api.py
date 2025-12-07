from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# Import the function and initialized objects from the new file
from bot_functions import run_qa, vectorstore, llm 

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # <-- IMPORTANT
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    k: int = 3

@app.post("/rag/query")
def generative_query(request: QueryRequest):
    """
    Performs generative search (RAG) against the indexed documents.
    """
    try:
        answer = run_qa(
            query=request.query,
            vectorstore=vectorstore,
            llm=llm,
            k=request.k
        )
        
        return {"query": request.query, "answer": answer}
        
    except Exception as e:
        return {"error": str(e), "message": "An error occurred during generative search."}

# Run this file with: uvicorn rag_api:app --reload