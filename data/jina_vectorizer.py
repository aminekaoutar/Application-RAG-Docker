"""
Jina AI Vectorization Module
Replaces sentence-transformers with Jina API for free embeddings
"""
import os
import json
import numpy as np
import requests
from typing import List, Dict, Any
import time

class JinaVectorizer:
    def __init__(self, api_key: str = None):
        """Initialize Jina vectorizer with API key"""
        self.api_key = api_key or os.getenv('JINA_API_KEY')
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable required")
        
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model = "jina-embeddings-v2-base-en"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for list of texts using Jina API"""
        if not texts:
            return []
        
        # Jina API has rate limits, so process in batches
        batch_size = 100  # Adjust based on your needs
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                payload = {
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float"
                }
                
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = [item['embedding'] for item in result['data']]
                    all_embeddings.extend(embeddings)
                    
                    # Rate limiting courtesy
                    if i + batch_size < len(texts):
                        time.sleep(0.1)
                        
                else:
                    print(f"Jina API error: {response.status_code} - {response.text}")
                    # Fallback: return zero vectors of appropriate dimension
                    dim = 768  # Jina base model dimension
                    all_embeddings.extend([[0.0] * dim for _ in batch])
                    
            except Exception as e:
                print(f"Error getting embeddings: {e}")
                # Fallback: return zero vectors
                dim = 768
                all_embeddings.extend([[0.0] * dim for _ in batch])
        
        return all_embeddings
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process documents and create embeddings"""
        # Extract text content
        texts = []
        doc_metadata = []
        
        for doc in documents:
            content = doc.get('text', '')  # Changed from 'content' to 'text'
            if content.strip():
                texts.append(content)
                doc_metadata.append({
                    'url': doc.get('url', ''),
                    'title': doc.get('title', ''),
                    'length': len(content)
                })
        
        if not texts:
            raise ValueError("No valid text content found in documents")
        
        print(f"Processing {len(texts)} documents...")
        
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        
        return {
            'texts': texts,
            'embeddings': embeddings,
            'metadata': doc_metadata,
            'model_used': self.model
        }

# Example usage function
def create_vector_store_with_jina(data_file: str = 'cleaned_data.json'):
    """Create vector store using Jina embeddings instead of sentence-transformers"""
    
    # Load cleaned data
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        print(f"Loaded {len(documents)} documents")
    except FileNotFoundError:
        print(f"Data file {data_file} not found")
        return None
    
    # Initialize Jina vectorizer
    try:
        vectorizer = JinaVectorizer()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set JINA_API_KEY environment variable")
        return None
    
    # Process documents
    try:
        result = vectorizer.process_documents(documents)
        print(f"Generated {len(result['embeddings'])} embeddings")
        
        # Save results (you can integrate with FAISS here)
        output_data = {
            'texts': result['texts'],
            'embeddings': result['embeddings'],
            'metadata': result['metadata'],
            'model': result['model_used'],
            'created_at': time.time()
        }
        
        with open('jina_vectors.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print("Vector store saved to jina_vectors.json")
        return output_data
        
    except Exception as e:
        print(f"Error processing documents: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    print("Jina Vectorization Pipeline")
    print("=" * 40)
    
    # Set your API key
    os.environ['JINA_API_KEY'] = 'jina_84de9166ea8d441fa7c3bb0a5c7bcc65Ae3bXjSqTXQI2yd-5K7tfbs7vtIn'
    
    # Run vectorization
    result = create_vector_store_with_jina()
    
    if result:
        print(f"\nSuccess! Generated {len(result['embeddings'])} vectors")
        print(f"Model used: {result['model']}")
        print(f"Average embedding dimension: {len(result['embeddings'][0])}")
    else:
        print("\nVectorization failed!")