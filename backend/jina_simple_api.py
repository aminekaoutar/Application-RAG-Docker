"""
Simple API using Jina-created FAISS index
"""
import os
import json
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: str

class HealthResponse(BaseModel):
    status: str
    vector_store_status: str

# Initialize FastAPI app
app = FastAPI(
    title="Web Scraping RAG API",
    description="API for question answering using scraped website data",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vector_store = None
texts = []
metadata = []

def load_vector_store():
    """Load the FAISS vector store created with Jina embeddings"""
    global vector_store, texts, metadata
    
    try:
        print("🚀 Loading FAISS vector store...")
        
        # Load FAISS index
        vector_store = faiss.read_index("digisand_vectors.faiss")
        print(f"✅ Loaded FAISS index with {vector_store.ntotal} vectors")
        
        # Load metadata
        with open("digisand_metadata.json", 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
            texts = meta_data['texts']
            metadata = meta_data['metadata']
        
        print(f"✅ Loaded {len(texts)} text documents")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return False

def get_jina_embedding(text: str) -> List[float]:
    """Get embedding for a text using Jina API"""
    try:
        api_key = os.getenv('JINA_API_KEY')
        if not api_key:
            raise ValueError("JINA_API_KEY not set")
        
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "jina-embeddings-v2-base-en",
            "input": [text]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['data'][0]['embedding']
        else:
            print(f"Jina API error: {response.status_code}")
            return [0.0] * 768  # Return zero vector as fallback
            
    except Exception as e:
        print(f"Error getting Jina embedding: {e}")
        return [0.0] * 768

@app.on_event("startup")
async def startup_event():
    """Initialize when server starts."""
    success = load_vector_store()
    if not success:
        print("⚠️  Vector store initialization failed")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    vector_status = "loaded" if vector_store else "not loaded"
    
    return HealthResponse(
        status="healthy",
        vector_store_status=vector_status
    )

@app.post("/query", response_model=QueryResponse)
async def answer_question(request: QueryRequest):
    """Answer questions using the vector store."""
    if not vector_store or not texts:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized"
        )
    
    try:
        print(f"❓ Processing query: {request.question}")
        
        # Get embedding for the question
        question_embedding = np.array([get_jina_embedding(request.question)]).astype('float32')
        
        # Search for similar vectors
        k = 3
        distances, indices = vector_store.search(question_embedding, k)
        
        # Get relevant texts
        relevant_texts = []
        sources = []
        
        for idx in indices[0]:
            if idx < len(texts):
                relevant_texts.append(texts[idx])
                if idx < len(metadata):
                    sources.append(metadata[idx].get('url', 'unknown'))
        
        if not relevant_texts:
            return QueryResponse(
                answer="I don't have specific information about that in the scraped data.",
                sources=[],
                conversation_id=request.conversation_id or "default"
            )
        
        # Simple answer generation (you can enhance this with Groq API)
        context = "\n\n".join(relevant_texts[:2])  # Use top 2 most relevant
        answer = f"Based on the scraped content, here's what I found:\n\n{context[:500]}..."
        
        print(f"✅ Generated answer for: {request.question}")
        
        return QueryResponse(
            answer=answer,
            sources=list(set(sources)),
            conversation_id=request.conversation_id or "default"
        )
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if not vector_store:
        return {"error": "Vector store not loaded"}
    
    return {
        "vector_count": vector_store.ntotal,
        "document_count": len(texts),
        "status": "ready"
    }

if __name__ == "__main__":
    uvicorn.run(
        "jina_simple_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )