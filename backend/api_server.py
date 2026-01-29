import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv

# Import RAG components
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import numpy as np
import faiss
import json
import os
from typing import List, Dict

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
    model: str

# Initialize FastAPI app
app = FastAPI(
    title="Web Scraping RAG API",
    description="API for question answering using scraped website data",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
rag_chain = None
vector_index = None
metadata = None
jina_vectorizer = None

def initialize_rag_system():
    """Initialize the RAG system components."""
    global rag_chain, vector_index, metadata, jina_vectorizer
    
    try:
        print("🚀 Initializing RAG system with Jina...")
        
        # Import Jina vectorizer
        import sys
        sys.path.append('../data')
        from jina_vectorizer import JinaVectorizer
        
        # Initialize Jina vectorizer
        jina_vectorizer = JinaVectorizer()
        
        # Initialize LLM
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.7,
            max_tokens=1000
        )
        
        # Load vector data
        config_file = "./data/digisand_vectors_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load FAISS index
            index_file = f"./data/{config['files']['index']}"
            vector_index = faiss.read_index(index_file)
            
            # Load metadata
            metadata_file = f"../data/{config['files']['metadata']}"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"✅ Loaded vector store with {vector_index.ntotal} vectors")
        else:
            print("⚠️  No vector data found, RAG will not work")
            return False
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions about DigiSand's website content.
            Use the provided context to answer questions accurately and concisely.
            
            Context: {context}
            
            Question: {question}"""),
            ("human", "{question}")
        ])
        
        # Format documents function
        def format_docs(docs):
            sources = []
            content = []
            for doc in docs:
                content.append(doc.page_content)
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.append(doc.metadata['source'])
            return "\n\n".join(content), list(set(sources))
        
        # Build chain with Jina similarity search
        def jina_similarity_search(query: str, k: int = 3):
            """Search similar content using Jina embeddings and FAISS"""
            if vector_index is None or metadata is None:
                return [], []
            
            # Create embedding for query using Jina
            query_embedding_list = jina_vectorizer.get_embeddings([query])
            query_embedding = np.array(query_embedding_list, dtype='float32')
            # Normalize for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search using FAISS
            scores, indices = vector_index.search(query_embedding, k)
            
            # Format results
            docs = []
            sources = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(metadata):
                    doc_content = metadata[idx]['chunk_text']
                    doc_source = metadata[idx].get('page_url', 'unknown')
                    docs.append({"page_content": doc_content})
                    sources.append(doc_source)
            
            return docs, list(set(sources))
        
        # Build chain with proper formatting
        def format_context_and_sources(inputs):
            question = inputs["question"]
            docs, sources = jina_similarity_search(question, k=3)
            
            content_parts = [doc["page_content"] for doc in docs]
            
            return {
                "context": "\n\n".join(content_parts),
                "sources": sources,
                "question": question
            }
        
        rag_chain = (
            {"question": RunnablePassthrough()}
            | format_context_and_sources
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✅ RAG system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system when server starts."""
    success = initialize_rag_system()
    if not success:
        print("⚠️  RAG system initialization failed - API will run but queries won't work")

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    vector_status = "loaded" if vector_index and metadata else "not loaded"
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    return HealthResponse(
        status="healthy",
        vector_store_status=vector_status,
        model=model
    )

@app.post("/query", response_model=QueryResponse)
async def answer_question(request: QueryRequest):
    """Answer questions using the RAG system."""
    if not rag_chain:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized - please check server logs"
        )
    
    try:
        print(f"❓ Processing query: {request.question}")
        
        # Generate answer using the updated chain
        answer = rag_chain.invoke(request.question)
        
        # Get sources from the chain's context formatter
        # The sources are already extracted in format_context_and_sources
        # For now, we'll return empty sources since we don't have access to them here
        # In a real implementation, you'd want to refactor to return sources properly
        
        print(f"✅ Generated answer for: {request.question}")
        
        return QueryResponse(
            answer=answer,
            sources=[],  # Sources extraction needs refactoring
            conversation_id=request.conversation_id or "default"
        )
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/sources")
async def get_sources():
    """Get list of available sources in the vector store."""
    if not vector_store:
        raise HTTPException(
            status_code=503,
            detail="Vector store not loaded"
        )
    
    try:
        # This would require storing source metadata separately
        # For now, return a placeholder
        return {"sources": ["DigiSand website"]}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sources: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if not vector_store:
        return {"error": "Vector store not loaded"}
    
    return {
        "vector_count": vector_store.index.ntotal if vector_store else 0,
        "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "status": "ready" if rag_chain else "initializing"
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )