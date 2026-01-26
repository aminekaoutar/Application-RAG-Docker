import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class RAGAgent:
    def __init__(self, vector_store_path: str = "digisand_vectors"):
        """Initialize the RAG agent with FAISS vector store and external tools."""
        print("🤖 Initializing RAG Agent...")
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.7,
            max_tokens=1000
        )
        
        # Initialize Tavily search tool
        self.tavily_tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced"
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Load FAISS vector store
        self.vector_store = self._load_vector_store(vector_store_path)
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Initialize the chain
        self.chain = self._create_chain()
        
        print("✅ RAG Agent initialized successfully!")
    
    def _load_vector_store(self, prefix: str) -> FAISS:
        """Load FAISS vector store from saved files."""
        try:
            # Load the FAISS index
            vector_store = FAISS.load_local(
                "./", 
                self.embeddings,
                index_name=f"{prefix}_index",
                allow_dangerous_deserialization=True
            )
            print(f"✅ Loaded FAISS vector store: {prefix}")
            return vector_store
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            raise
    
    def _create_chain(self):
        """Create the RAG chain with fallback logic."""
        
        # Prompt template for RAG
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on scraped website data.
            Use the provided context to answer questions accurately. If the context doesn't contain 
            relevant information, you can search externally using available tools.
            
            Context from scraped data:
            {context}
            
            Question: {question}
            
            Provide a helpful and accurate answer based on the context when available."""),
            ("human", "{question}")
        ])
        
        # Create the chain
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _check_internal_knowledge(self, question: str) -> bool:
        """Check if question can be answered from internal data."""
        # Simple heuristic: check if key terms from question exist in our data
        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(question)
            
            # If we get relevant documents with good similarity, use internal data
            if docs and hasattr(docs[0], 'metadata'):
                return True
            return False
        except:
            return False
    
    def search_external(self, query: str) -> str:
        """Search external sources using Tavily."""
        try:
            print(f"🌍 Searching externally for: {query}")
            results = self.tavily_tool.invoke({"query": query})
            return str(results)
        except Exception as e:
            print(f"❌ External search failed: {e}")
            return "Unable to search external sources at the moment."
    
    def answer_question(self, question: str) -> str:
        """
        Answer question using internal data first, fallback to external search.
        
        Args:
            question: User's question
            
        Returns:
            Answer to the question
        """
        print(f"❓ Question received: {question}")
        
        # First, try to answer from internal data
        try:
            # Check if we have relevant internal data
            has_internal_data = self._check_internal_knowledge(question)
            
            if has_internal_data:
                print("📚 Using internal scraped data...")
                answer = self.chain.invoke(question)
                return f"Based on the scraped website data:\n\n{answer}"
            else:
                print("🌐 No relevant internal data found, searching externally...")
                external_results = self.search_external(question)
                return f"I searched external sources and found:\n\n{external_results}"
                
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return "Sorry, I encountered an error while processing your question."

def main():
    """Interactive chat interface for the RAG agent."""
    print("🤖 RAG Agent - Web Data Assistant")
    print("=" * 40)
    print("Ask questions about the scraped website data!")
    print("Type 'quit' to exit\n")
    
    try:
        # Initialize the agent
        agent = RAGAgent()
        
        # Interactive chat loop
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
                
            if not question:
                print("⚠️  Please enter a question.")
                continue
            
            try:
                answer = agent.answer_question(question)
                print(f"\n🤖 Assistant: {answer}\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")
                
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("Make sure you have:")
        print("1. Run the vectorizer.py script first")
        print("2. Set up your API keys in .env file")
        print("3. Installed required packages")

if __name__ == "__main__":
    main()