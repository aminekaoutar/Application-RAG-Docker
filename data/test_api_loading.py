import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Test what files exist and try different loading approaches
print("Checking vector files...")
files = [f for f in os.listdir('.') if 'digisand' in f]
for f in files:
    print(f"- {f}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Try different index names
possible_names = [
    "digisand_vectors_index",
    "digisand_vectors", 
    "index"
]

for name in possible_names:
    try:
        print(f"\nTrying to load: {name}")
        vector_store = FAISS.load_local(
            "./",
            embeddings,
            index_name=name,
            allow_dangerous_deserialization=True
        )
        print(f"✅ SUCCESS! Loaded {vector_store.index.ntotal} vectors")
        break
    except Exception as e:
        print(f"❌ Failed: {e}")