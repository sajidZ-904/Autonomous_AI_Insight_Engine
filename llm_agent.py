import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMAgent:
    def __init__(self):
        # Initialize OpenAI API key from environment variable
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
    
    def generate_answer(self, query, relevant_docs):
        """Generate a structured answer using the retrieved documents"""
        # Prepare context from relevant documents
        context = "\n\n".join([
            f"Document {doc['id']}: {doc['content']}"
            for doc in relevant_docs
        ])
        
        # Create the prompt
        prompt = f"""Based ONLY on the following documents, provide a structured answer to the query.
Do not make up any information that is not in the documents.
Format the response with bullet points and include source citations [Source: doc_id].

Documents:
{context}

Query: {query}

Answer:"""
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides accurate, source-cited answers based only on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=500
            )
            
            # Extract the answer
            answer = response.choices[0].message.content
            
            # Return structured response
            return {
                "answer": answer,
                "sources": [doc["id"] for doc in relevant_docs]
            }
            
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
