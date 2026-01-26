import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

class SimpleRAGAgent:
    def __init__(self):
        """Simple RAG agent with known working configuration."""
        print("🤖 Initializing Simple RAG Agent...")
        
        # Initialize components
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.7,
            max_tokens=1000
        )
        
        # Load vector store (known to work)
        self.vector_store = FAISS.load_local(
            "./",
            self.embeddings,
            index_name="digisand_vectors",
            allow_dangerous_deserialization=True
        )
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Create chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions about a company's website.
            Use the provided context to answer questions accurately.
            
            Context: {context}
            Question: {question}"""),
            ("human", "{question}")
        ])
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ Simple RAG Agent initialized successfully!")
        print(f"📊 Vector store contains {self.vector_store.index.ntotal} vectors")
    
    def answer_question(self, question: str) -> str:
        """Answer a question using the RAG system."""
        try:
            print(f"❓ Processing: {question}")
            
            # Get relevant documents
            docs = self.vector_store.similarity_search(question, k=3)
            
            if docs:
                print("📚 Found relevant internal data")
                answer = self.chain.invoke(question)
                return f"Based on the scraped website data:\n\n{answer}"
            else:
                print("🌐 No relevant internal data, suggesting external search")
                return "I don't have specific information about that in the scraped data. You might want to search externally for this topic."
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return "Sorry, I encountered an error processing your question."

def main():
    """Simple interactive interface."""
    print("🤖 Simple RAG Agent")
    print("=" * 30)
    print("Ask questions about the scraped website!")
    print("Type 'quit' to exit\n")
    
    try:
        agent = SimpleRAGAgent()
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
            
            answer = agent.answer_question(question)
            print(f"\n🤖 Assistant: {answer}\n")
            
    except Exception as e:
        print(f"❌ Initialization failed: {e}")

if __name__ == "__main__":
    main()