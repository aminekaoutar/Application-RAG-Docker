import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Tuple
import time

class DataVectorizer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vectorizer with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
                       'all-MiniLM-L6-v2' is fast, lightweight, and free
        """
        print(f"🚀 Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Embedding dimension: {self.dimension}")
        
        # FAISS index for vector storage
        self.index = None
        self.metadata = []  # Store URL and other info
        
    def load_data(self, filepath: str) -> List[Dict]:
        """Load cleaned data from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return []
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better vectorization.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in characters (reduced for memory)
            overlap: Number of characters to overlap between chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        # Limit total chunks to prevent memory issues
        max_chunks = 50
        chunk_count = 0
        
        while start < len(text) and chunk_count < max_chunks:
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            chunk_count += 1
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
                
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create vector embeddings for a list of texts."""
        print(f"🧠 Creating embeddings for {len(texts)} text chunks...")
        start_time = time.time()
        
        # Generate embeddings with batch processing for memory efficiency
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Important for FAISS cosine similarity
            batch_size=32  # Reduce batch size for memory efficiency
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Embeddings created in {elapsed:.2f} seconds")
        print(f"📊 Embedding shape: {embeddings.shape}")
        
        return embeddings.astype('float32')  # FAISS requires float32
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for efficient similarity search."""
        print("🏗️  Building FAISS index...")
        start_time = time.time()
        
        # Use IndexFlatIP for cosine similarity (since we normalized embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        elapsed = time.time() - start_time
        print(f"✅ FAISS index built in {elapsed:.2f} seconds")
        print(f"📊 Index contains {self.index.ntotal} vectors")
    
    def vectorize_data(self, 
                      input_file: str = "cleaned_data.json",
                      output_prefix: str = "vectorized_data") -> Tuple[np.ndarray, List[Dict]]:
        """
        Main vectorization function.
        
        Args:
            input_file: Path to cleaned JSON data
            output_prefix: Prefix for output files
            
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        print("🔬 Starting data vectorization process...")
        print("=" * 50)
        
        # Load data
        data = self.load_data(input_file)
        if not data:
            print("❌ No data to vectorize. Exiting.")
            return None, []
        
        print(f"📄 Loaded {len(data)} pages")
        
        # Prepare data for vectorization
        all_chunks = []
        metadata = []
        
        print("📝 Chunking text data...")
        for page_idx, page in enumerate(data):
            text = page.get('text', '')
            url = page.get('url', f'page_{page_idx}')
            
            if not text.strip():
                print(f"⚠️  Skipping empty page: {url}")
                continue
            
            # Split text into chunks
            chunks = self.chunk_text(text, chunk_size=500, overlap=100)
            
            # Store chunks with metadata
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    'page_url': url,
                    'page_index': page_idx,
                    'chunk_index': chunk_idx,
                    'chunk_text': chunk[:100] + '...' if len(chunk) > 100 else chunk,
                    'full_text_length': len(text)
                })
            
            print(f"   Page {page_idx + 1}: {url} -> {len(chunks)} chunks")
        
        print(f"📊 Total chunks to vectorize: {len(all_chunks)}")
        
        # Create embeddings
        embeddings = self.create_embeddings(all_chunks)
        
        # Build FAISS index
        self.build_faiss_index(embeddings)
        
        # Save vectorized data
        self.save_vectorized_data(output_prefix, embeddings, metadata)
        
        return embeddings, metadata
    
    def save_vectorized_data(self, prefix: str, embeddings: np.ndarray, metadata: List[Dict]):
        """Save vectorized data and FAISS index."""
        print("💾 Saving vectorized data...")
        
        # Save FAISS index
        index_file = f"{prefix}_index.faiss"
        faiss.write_index(self.index, index_file)
        print(f"✅ FAISS index saved to: {index_file}")
        
        # Save metadata
        metadata_file = f"{prefix}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ Metadata saved to: {metadata_file}")
        
        # Save embeddings (optional - large file)
        embeddings_file = f"{prefix}_embeddings.npy"
        np.save(embeddings_file, embeddings)
        print(f"✅ Embeddings saved to: {embeddings_file}")
        
        # Save configuration
        config = {
            'dimension': self.dimension,
            'model_name': 'all-MiniLM-L6-v2',  # Hardcoded since get_model_name() doesn't exist
            'total_vectors': len(embeddings),
            'total_chunks': len(metadata),
            'files': {
                'index': index_file,
                'metadata': metadata_file,
                'embeddings': embeddings_file
            }
        }
        
        config_file = f"{prefix}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration saved to: {config_file}")
    
    def load_vectorized_data(self, prefix: str):
        """Load previously vectorized data."""
        print("📂 Loading vectorized data...")
        
        # Load configuration
        config_file = f"{prefix}_config.json"
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        # Load FAISS index
        index_file = config['files']['index']
        try:
            self.index = faiss.read_index(index_file)
            self.dimension = config['dimension']
            print(f"✅ FAISS index loaded: {index_file}")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            return False
        
        # Load metadata
        metadata_file = config['files']['metadata']
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"✅ Metadata loaded: {metadata_file}")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return False
        
        print(f"📊 Loaded {self.index.ntotal} vectors")
        return True
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar content using vector similarity.
        
        Args:
            query: Search query text
            k: Number of similar results to return
            
        Returns:
            List of similar items with scores and metadata
        """
        if self.index is None:
            print("❌ No vector index loaded. Please load data first.")
            return []
        
        # Create embedding for query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search using FAISS
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):  # Valid index
                result = {
                    'rank': i + 1,
                    'similarity_score': float(score),
                    'metadata': self.metadata[idx],
                    'chunk_preview': self.metadata[idx]['chunk_text']
                }
                results.append(result)
        
        return results

def main():
    """Main function to demonstrate vectorization."""
    # Initialize vectorizer
    vectorizer = DataVectorizer()
    
    # Vectorize the cleaned data
    embeddings, metadata = vectorizer.vectorize_data(
        input_file="cleaned_data.json",
        output_prefix="digisand_vectors"
    )
    
    if embeddings is not None:
        print(f"\n🎉 Vectorization completed!")
        print(f"📊 Total vectors: {len(embeddings)}")
        print(f"📁 Output files created with prefix: digisand_vectors")
        
        # Demonstrate search functionality
        print(f"\n🔍 Demo search:")
        query = "digital marketing services"
        results = vectorizer.search_similar(query, k=3)
        
        print(f"Query: '{query}'")
        print("-" * 50)
        for result in results:
            print(f"Rank {result['rank']} (Score: {result['similarity_score']:.3f})")
            print(f"URL: {result['metadata']['page_url']}")
            print(f"Preview: {result['chunk_preview']}")
            print()

if __name__ == "__main__":
    main()