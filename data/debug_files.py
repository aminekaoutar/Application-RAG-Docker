import os

# List all files with detailed info
print("All files in directory:")
for file in os.listdir('.'):
    if 'digisand' in file:
        size = os.path.getsize(file)
        print(f"{file:<30} {size:>10} bytes")

print("\nTrying manual load...")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    # Try direct loading
    vector_store = FAISS.load_local(
        "./", 
        embeddings,
        index_name="digisand_vectors",
        allow_dangerous_deserialization=True
    )
    print(f"✅ SUCCESS! Loaded {vector_store.index.ntotal} vectors")
except Exception as e:
    print(f"❌ Failed: {e}")
    
    # Try alternative names
    alternatives = ["digisand", "index"]
    for alt in alternatives:
        try:
            vector_store = FAISS.load_local(
                "./", 
                embeddings,
                index_name=alt,
                allow_dangerous_deserialization=True
            )
            print(f"✅ SUCCESS with {alt}! Loaded {vector_store.index.ntotal} vectors")
            break
        except Exception as e2:
            print(f"❌ {alt} failed: {e2}")