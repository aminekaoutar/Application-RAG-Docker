"""
Convert Jina vectors to FAISS format for backend compatibility
"""
import json
import numpy as np
import faiss

def convert_jina_to_faiss(jina_file: str = 'jina_vectors.json', output_prefix: str = 'digisand'):
    """Convert Jina API vectors to FAISS format"""
    
    # Load Jina vectors
    with open(jina_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    embeddings = np.array(data['embeddings']).astype('float32')
    texts = data['texts']
    metadata = data['metadata']
    
    print(f"Loaded {len(embeddings)} vectors of dimension {embeddings.shape[1]}")
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, f'{output_prefix}_vectors.faiss')
    
    # Save metadata
    metadata_dict = {
        'texts': texts,
        'metadata': metadata,
        'dimension': dimension,
        'count': len(embeddings)
    }
    
    with open(f'{output_prefix}_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ FAISS index saved as {output_prefix}_vectors.faiss")
    print(f"✅ Metadata saved as {output_prefix}_metadata.json")
    print(f"📊 Index contains {index.ntotal} vectors")

if __name__ == "__main__":
    convert_jina_to_faiss()