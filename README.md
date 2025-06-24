# AI Insight Engine

An autonomous LLM-powered intelligence engine that retrieves context from a knowledge base and generates structured, source-cited responses.

## Features

- FastAPI endpoint for query processing
- Embedding-based document retrieval using FAISS
- LLM-powered answer generation with OpenAI GPT-3.5
- Hallucination mitigation through context-grounded responses
- Source citation for transparency

## Project Structure

```
.
├── data/          # Knowledge base documents
├── main.py        # FastAPI application
├── retriever.py   # Vector store & retrieval logic
├── llm_agent.py   # LLM prompt & response handling
└── requirements.txt
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Usage

Send a POST request to `/query` with your research query:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the top 3 use cases for GraphQL in enterprise SaaS?"}'
```

Example Response:
```json
{
    "answer": "Based on the provided documents, here are the top 3 use cases for GraphQL in enterprise SaaS:\n\n• Flexible API Integration: GraphQL enables clients to request exactly the data they need, eliminating over-fetching and under-fetching common in REST APIs. This is particularly valuable for complex, nested data structures in enterprise applications. [Source: doc_01]\n\n• Real-time Data Updates: GraphQL subscriptions facilitate real-time data synchronization, which is essential for dashboards, monitoring tools, and collaborative features where immediate data updates are crucial. [Source: doc_02]\n\n• Performance Optimization: GraphQL's ability to batch multiple queries into a single request reduces network overhead and server load, which is particularly important for mobile applications and distributed systems. [Source: doc_03]",
    "sources": ["doc_01", "doc_02", "doc_03"]
}
```

## Sample Documents

The system comes with 10 sample documents in the knowledge base, covering topics like GraphQL, AI/ML, cloud computing, and enterprise architecture. You can add your own documents by placing JSON files in the `data/` directory following the same format as the sample documents.
