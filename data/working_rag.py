import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

def create_rag_chain():
    """Create a working RAG chain."""
    print("🤖 Setting up RAG system...")
    
    # Initialize components
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
    
    # Create the chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant answering questions about DigiSand's website content.
        Use the provided context to answer questions accurately and concisely.
        
        Context: {context}
        
        Question: {question}"""),
        ("human", "{question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Build chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, vector_store

def main():
    """Interactive RAG interface."""
    print("🤖 Website RAG Assistant")
    print("=" * 35)
    print("Ask questions about the scraped website!")
    print("Type 'quit' to exit\n")
    
    try:
        chain, vector_store = create_rag_chain()
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
            
            try:
                print("🤔 Processing your question...")
                answer = chain.invoke(question)
                print(f"\n{answer}\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")
                
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("Make sure you've run the vectorization process first.")

if __name__ == "__main__":
    main()