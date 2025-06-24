from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import DocumentRetriever
from llm_agent import LLMAgent
import json

app = FastAPI(title="AI Insight Engine")

# Initialize our components
retriever = DocumentRetriever()
llm_agent = LLMAgent()

class Query(BaseModel):
    query: str

@app.post("/query")
async def process_query(query: Query):
    try:
        # Retrieve relevant documents
        relevant_docs = retriever.retrieve(query.query)
        
        # Generate answer using LLM
        response = llm_agent.generate_answer(query.query, relevant_docs)
        
        return {
            "answer": response["answer"],
            "sources": response["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
