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
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
vector_store = None
embeddings = None

def initialize_rag_system():
    """Initialize the RAG system components."""
    global rag_chain, vector_store, embeddings
    
    try:
        print("🚀 Initializing RAG system...")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize LLM
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.7,
            max_tokens=1000
        )
        
        # Load vector store
        vector_store = FAISS.load_local(
            "./",
            embeddings,
            index_name="digisand_vectors",
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ Loaded vector store with {vector_store.index.ntotal} vectors")
        
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
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Build chain with proper formatting
        def format_context_and_sources(inputs):
            docs = inputs["docs"]
            sources = []
            content_parts = []
            
            for doc in docs:
                content_parts.append(doc.page_content)
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.append(doc.metadata['source'])
            
            return {
                "context": "\n\n".join(content_parts),
                "sources": list(set(sources)),
                "question": inputs["question"]
            }
        
        rag_chain = (
            {"docs": retriever, "question": RunnablePassthrough()}
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
    vector_status = "loaded" if vector_store else "not loaded"
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
        
        # Get relevant documents and sources
        docs = vector_store.similarity_search(request.question, k=3)
        
        if not docs:
            return QueryResponse(
                answer="I don't have specific information about that in the scraped data.",
                sources=[],
                conversation_id=request.conversation_id or "default"
            )
        
        # Extract sources
        sources = []
        for doc in docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.append(doc.metadata['source'])
        
        # Generate answer
        answer = rag_chain.invoke(request.question)
        
        print(f"✅ Generated answer for: {request.question}")
        
        return QueryResponse(
            answer=answer,
            sources=list(set(sources)),  # Remove duplicates
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