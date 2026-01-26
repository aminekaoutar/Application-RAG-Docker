#!/usr/bin/env python3
"""
Simple search interface for vectorized data
"""

from vectorizer import DataVectorizer

def main():
    """Interactive search interface."""
    print("🔍 Vector Search Interface")
    print("=" * 30)
    
    # Load the vectorized data
    vectorizer = DataVectorizer()
    
    if not vectorizer.load_vectorized_data("digisand_vectors"):
        print("❌ Failed to load vectorized data. Please run vectorizer.py first.")
        return
    
    print("✅ Vector data loaded successfully!")
    print(f"📊 Available vectors: {vectorizer.index.ntotal}")
    print()
    
    # Interactive search loop
    while True:
        query = input("Enter search query (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            print("👋 Goodbye!")
            break
            
        if not query:
            print("⚠️  Please enter a query.")
            continue
        
        try:
            # Perform search
            results = vectorizer.search_similar(query, k=5)
            
            if not results:
                print("❌ No results found.")
                continue
            
            print(f"\n🎯 Results for: '{query}'")
            print("-" * 50)
            
            for result in results:
                print(f"Rank {result['rank']} (Score: {result['similarity_score']:.3f})")
                print(f"🔗 URL: {result['metadata']['page_url']}")
                print(f"📋 Preview: {result['chunk_preview']}")
                print()
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            continue

if __name__ == "__main__":
    main()