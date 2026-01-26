import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

def recreate_vector_store():
    """Recreate the FAISS vector store in LangChain format."""
    
    print("🔄 Recreating vector store in LangChain format...")
    
    # Load the original data
    try:
        with open("cleaned_data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} pages from cleaned_data.json")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Prepare documents
    documents = []
    
    for page in data:
        text = page.get('text', '')
        url = page.get('url', '')
        
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={
                    'source': url,
                    'title': page.get('title', ''),
                    'original_length': page.get('original_length', 0),
                    'cleaned_length': page.get('cleaned_length', 0)
                }
            )
            documents.append(doc)
    
    print(f"📊 Prepared {len(documents)} documents")
    
    # Create FAISS vector store
    try:
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        print("✅ Created FAISS vector store")
        
        # Save in LangChain format
        vector_store.save_local("./", "digisand_vectors")
        print("✅ Saved vector store in LangChain format")
        
        # Test loading
        loaded_store = FAISS.load_local(
            "./",
            embeddings,
            index_name="digisand_vectors",
            allow_dangerous_deserialization=True
        )
        print(f"✅ Successfully tested loading: {loaded_store.index.ntotal} vectors")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return False

if __name__ == "__main__":
    success = recreate_vector_store()
    if success:
        print("\n🎉 Vector store recreation completed!")
        print("Now you can run the RAG agent:")
        print("python rag_agent.py")
    else:
        print("\n❌ Vector store recreation failed!")