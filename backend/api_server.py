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

class ScrapeRequest(BaseModel):
    url: str
    max_pages: Optional[int] = 10

class ScrapeResponse(BaseModel):
    status: str
    pages_scraped: int
    message: str

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
scraped_data = []  # Store dynamically scraped content

# Global function for RAG invocation
def rag_invoke_with_sources(question: str):
    """Invoke RAG chain and return both answer and sources"""
    global vector_index, metadata, jina_vectorizer
    
    if vector_index is None or metadata is None or jina_vectorizer is None:
        return "RAG system not ready", []
    
    # Create embedding for query using Jina
    query_embedding_list = jina_vectorizer.get_embeddings([question])
    query_embedding = np.array(query_embedding_list, dtype='float32')
    # Normalize for cosine similarity
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Search using FAISS
    scores, indices = vector_index.search(query_embedding, 3)
    
    # Format results
    content_parts = []
    sources = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(metadata):
            doc_content = metadata[idx]['chunk_text']
            doc_source = metadata[idx].get('page_url', 'unknown')
            content_parts.append(doc_content)
            sources.append(doc_source)
    
    context = "\n\n".join(content_parts)
    
    # Create prompt with context
    full_prompt = f"""You are a helpful assistant answering questions about website content.
Use the provided context to answer questions accurately and concisely.

Context: {context}

Question: {question}"""
    
    # Get answer from LLM
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.7,
        max_tokens=1000
    )
    answer = llm.invoke(full_prompt)
    
    return str(answer), sources

def initialize_rag_system():
    """Initialize the RAG system components."""
    global rag_chain, vector_index, metadata, jina_vectorizer
    
    try:
        print("🚀 Initializing RAG system with Jina...")
        
        # Import Jina vectorizer
        import sys
        sys.path.append('/app/data')
        from jina_vectorizer import JinaVectorizer
        
        # Initialize Jina vectorizer
        jina_vectorizer = JinaVectorizer()
        
        # Initialize LLM
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.7,
            max_tokens=1000
        )
        
        # Try to load existing vector data first
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
            
            print(f"✅ Loaded existing vector store with {vector_index.ntotal} vectors")
        else:
            print("⚠️  No existing vector data found")
            # System can still work if user scrapes new data
            return True
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions about DigiSand's website content.
            Use the provided context to answer questions accurately and concisely.
            
            Context: {context}
            
            Question: {question}"""),
            ("human", "{question}")
        ])
        
        # Store the invoke function globally
        global rag_chain
        rag_chain = rag_invoke_with_sources
        
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
    
    if vector_index is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail="No vector data loaded - please scrape a website first"
        )
    
    try:
        print(f"❓ Processing query: {request.question}")
        
        # Generate answer and get sources
        answer, sources = rag_chain(request.question)
        
        print(f"✅ Generated answer for: {request.question}")
        print(f"🔗 Sources: {sources}")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
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
    if not metadata:
        raise HTTPException(
            status_code=503,
            detail="No metadata loaded"
        )
    
    try:
        # Extract unique sources from metadata
        sources = list(set([item.get('page_url', 'unknown') for item in metadata]))
        return {"sources": sources[:50]}  # Limit to first 50 sources
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sources: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "vector_count": vector_index.ntotal if vector_index else 0,
        "sources_count": len(metadata) if metadata else 0,
        "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "status": "ready" if rag_chain and vector_index else "needs_data"
    }

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_website(request: ScrapeRequest):
    """Scrape a website and create vector embeddings for RAG."""
    global vector_index, metadata, jina_vectorizer
    
    try:
        print(f"🕷️  Starting scrape for: {request.url}")
        
        # Import scraper components
        import sys
        sys.path.append('.')
        from simple_scraper import scrape_single_page, clean_scraped_text
        from data.vectorizer import DataVectorizer
        
        # Scrape the website
        scraped_data = scrape_single_page(request.url)
        
        if not scraped_data["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to scrape website: {scraped_data['error']}"
            )
        
        print(f"✅ Successfully scraped: {scraped_data['url']}")
        print(f"📝 Content length: {len(scraped_data['text'])} characters")
        
        # Save scraped data to temporary file
        temp_file = "temp_scraped_data.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump([scraped_data], f, ensure_ascii=False, indent=2)
        
        # Create vectorizer and process the data
        print("🧠 Creating embeddings...")
        vectorizer = DataVectorizer(use_jina=True)
        
        # Vectorize the scraped data
        embeddings, new_metadata = vectorizer.vectorize_data(
            input_file=temp_file,
            output_prefix="dynamic_vectors"
        )
        
        if embeddings is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to create embeddings from scraped data"
            )
        
        # Update global variables
        global vector_index, metadata, rag_chain
        vector_index = vectorizer.index
        metadata = new_metadata
        
        # Rebuild the RAG chain with new data
        rag_chain = rag_invoke_with_sources
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print(f"✅ Vectorization complete. Indexed {len(embeddings)} chunks.")
        
        # Reinitialize RAG chain with new data
        # Note: We don't need to reinitialize since we updated global variables directly
        print("✅ RAG system updated with new data")
        
        return ScrapeResponse(
            status="completed",
            pages_scraped=1,
            message=f"Successfully scraped and indexed {request.url}. Ready for queries."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error scraping website: {str(e)}"
        )

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )