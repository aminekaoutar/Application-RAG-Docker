import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Test loading with explicit file names
def test_vector_loading():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # List all .faiss files
    faiss_files = [f for f in os.listdir('.') if f.endswith('.faiss')]
    print("FAISS files found:", faiss_files)
    
    # Try to load each one
    for faiss_file in faiss_files:
        index_name = faiss_file.replace('_index.faiss', '').replace('.faiss', '')
        print(f"\nTrying to load: {index_name}")
        
        try:
            vector_store = FAISS.load_local(
                "./",
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Success! Loaded {vector_store.index.ntotal} vectors")
            return vector_store, index_name
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return None, None

# Run the test
vector_store, index_name = test_vector_loading()
if vector_store:
    print(f"\n🎉 Successfully loaded vector store: {index_name}")
    print("Ready to use in RAG agent!")
else:
    print("\n❌ Could not load any vector store")