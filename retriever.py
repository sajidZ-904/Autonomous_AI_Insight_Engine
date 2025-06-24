import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class DocumentRetriever:
    def __init__(self):
        # Initialize the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load and embed documents
        self.documents = self._load_documents()
        self.document_embeddings = self._create_embeddings()
        
        # Create FAISS index
        self.index = self._create_index()
    
    def _load_documents(self):
        """Load documents from the data directory"""
        documents = []
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        # Load sample documents if they don't exist
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            self._create_sample_documents()
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    doc = json.load(f)
                    documents.append(doc)
        
        return documents

    def _create_embeddings(self):
        """Create embeddings for all documents"""
        texts = [doc['content'] for doc in self.documents]
        return self.model.encode(texts)
    
    def _create_index(self):
        """Create FAISS index for fast similarity search"""
        dimension = self.document_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.document_embeddings.astype('float32'))
        return index
    
    def retrieve(self, query, top_k=3):
        """Retrieve most relevant documents for a query"""
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return relevant documents
        relevant_docs = []
        for idx in indices[0]:
            doc = self.documents[idx]
            relevant_docs.append({
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content']
            })
        
        return relevant_docs
    
    def _create_sample_documents(self):
        """Create sample documents for testing"""
        sample_docs = [
            {
                "id": "doc_01",
                "title": "GraphQL in Enterprise: API Integration",
                "content": "GraphQL excels in enterprise SaaS by enabling flexible API integration. It allows clients to request exactly the data they need, reducing over-fetching and under-fetching common in REST APIs. This is particularly valuable when dealing with complex, nested data structures typical in enterprise applications."
            },
            {
                "id": "doc_02",
                "title": "GraphQL: Real-time Data Updates",
                "content": "GraphQL subscriptions provide real-time data updates in enterprise SaaS applications. This feature is crucial for dashboards, monitoring tools, and collaborative features where users need immediate data synchronization. The subscription mechanism ensures efficient real-time communication between server and clients."
            },
            {
                "id": "doc_03",
                "title": "GraphQL Performance Optimization",
                "content": "In enterprise SaaS, GraphQL's ability to batch multiple queries into a single request significantly improves performance. This reduces network overhead and server load, especially important for mobile applications or distributed systems where bandwidth efficiency is crucial."
            },
            {
                "id": "doc_04",
                "title": "AI and Machine Learning Overview",
                "content": "AI and machine learning are transforming enterprise software. From predictive analytics to natural language processing, these technologies enable automation and intelligent decision-making at scale."
            },
            {
                "id": "doc_05",
                "title": "Cloud Computing Fundamentals",
                "content": "Cloud computing provides scalable, flexible infrastructure for modern applications. Key concepts include IaaS, PaaS, and SaaS models, each serving different enterprise needs."
            },
            {
                "id": "doc_06",
                "title": "Microservices Architecture",
                "content": "Microservices architecture breaks down complex applications into smaller, independent services. This approach improves scalability, maintainability, and deployment flexibility."
            },
            {
                "id": "doc_07",
                "title": "DevOps Best Practices",
                "content": "DevOps combines development and operations to streamline software delivery. Key practices include continuous integration, continuous deployment, and automated testing."
            },
            {
                "id": "doc_08",
                "title": "Data Security in Enterprise",
                "content": "Enterprise data security encompasses encryption, access control, and compliance measures. Regular security audits and updates are essential for maintaining data integrity."
            },
            {
                "id": "doc_09",
                "title": "API Design Patterns",
                "content": "Effective API design follows RESTful principles, ensures versioning, and provides comprehensive documentation. These patterns promote API adoption and maintainability."
            },
            {
                "id": "doc_10",
                "title": "Scalable Database Systems",
                "content": "Modern database systems must handle increasing data volumes while maintaining performance. Solutions include sharding, replication, and optimized indexing strategies."
            }
        ]
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save sample documents
        for doc in sample_docs:
            filename = f"{doc['id']}.json"
            with open(os.path.join(data_dir, filename), 'w') as f:
                json.dump(doc, f, indent=2)
