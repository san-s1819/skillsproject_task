"""
Agent Module for Automotive Recall Agent
Orchestrates RAG pipeline: retrieval + generation
"""

from typing import Optional
from .vector_store import VectorStore
from .llm_client import GeminiClient
from .logger import QueryLogger
import os
import dotenv

dotenv.load_dotenv()


class RecallAgent:
    """Main agent class orchestrating RAG pipeline for recall queries"""
    
    def __init__(
        self, 
        data_dir: str = "data/recalls",
        api_key: str = os.getenv('GEMINI_API_KEY'),
        top_k: int = 3,
        enable_logging: bool = True,
        log_file: str = "logs/query_log.txt"
    ):
        """
        Initialize Recall Agent
        
        Args:
            data_dir: Directory containing recall documents
            api_key: Gemini API key (optional, reads from env if not provided)
            top_k: Number of documents to retrieve for each query
            enable_logging: Whether to enable query logging
            log_file: Path to log file
        """
        self.top_k = top_k
        self.enable_logging = enable_logging
        
        # Initialize logger
        if enable_logging:
            self.logger = QueryLogger(log_file=log_file)
        else:
            self.logger = None
        
        print("Initializing Automotive Recall Agent...")
        print("=" * 60)
        
        # Initialize vector store
        print("\n[1/2] Setting up vector store...")
        try:
            self.vector_store = VectorStore(data_dir=data_dir)
            self.vector_store.initialize()
        except Exception as e:
            print(f"[RecallAgent.__init__] Error initializing vector store: {type(e).__name__}: {e}")
            raise
        
        # Initialize LLM client
        print("\n[2/2] Setting up Gemini API client...")
        try:
            self.llm_client = GeminiClient(api_key=api_key)
        except Exception as e:
            print(f"[RecallAgent.__init__] Error initializing LLM client: {type(e).__name__}: {e}")
            raise
        
        print("\n" + "=" * 60)
        print("✓ Agent initialization complete!")
        print(f"Ready to answer recall queries (retrieving top-{top_k} documents)\n")
    
    def query(self, user_input: str) -> str:
        """
        Process a user query through the RAG pipeline
        
        Args:
            user_input: User's question about recalls
            
        Returns:
            Generated response from the agent
        """
        try:
            # Step 1: Retrieve relevant documents with confidence scoring
            print(f"\n🔍 Retrieving relevant recall documents...")
            try:
                retrieval_result = self.vector_store.retrieve_documents(
                    user_input, 
                    top_k=self.top_k
                )
            except Exception as e:
                print(f"[RecallAgent.query] Error during retrieval: {type(e).__name__}: {e}")
                raise
            
            retrieved_docs = retrieval_result['documents']
            confidence = retrieval_result['confidence']
            
            if not retrieved_docs:
                return (
                    "I couldn't find any relevant recall information for your query. "
                    "Please try rephrasing your question or ask about a specific vehicle make and model."
                )
            
            # Show query expansion info
            if retrieval_result['query_expanded']:
                print(f"   📝 Expanded query: '{retrieval_result['expanded_query']}'")
            
            # Show retrieved documents
            print(f"   Retrieved {len(retrieved_docs)} documents:")
            for doc_result in retrieved_docs:
                doc = doc_result['document']
                score = doc_result['similarity_score']
                print(f"   - {doc['filename']} (relevance: {score:.2f})")
            
            # Show confidence score
            confidence_emoji = {"HIGH": "✅", "MEDIUM": "⚠️", "LOW": "❌"}
            emoji = confidence_emoji.get(confidence['level'], "❓")
            print(f"   {emoji} Confidence: {confidence['level']} - {confidence['reason']}")
            
            # Step 2: Generate prompt with citations
            try:
                prompt = self.llm_client.create_recall_prompt(user_input, retrieved_docs)
            except Exception as e:
                print(f"[RecallAgent.query] Error creating prompt: {type(e).__name__}: {e}")
                raise
            
            # Step 3: Generate response
            print(f"\n🤖 Generating response with Gemini...")
            try:
                response = self.llm_client.generate_response(prompt)
            except Exception as e:
                print(f"[RecallAgent.query] Error generating response: {type(e).__name__}: {e}")
                raise
            
            # Log the query interaction
            if self.enable_logging and self.logger:
                try:
                    self.logger.log_query(
                        user_query=user_input,
                        retrieved_chunks=retrieved_docs,
                        confidence=confidence,
                        llm_response=response,
                        query_expanded=retrieval_result['query_expanded'],
                        expanded_query=retrieval_result.get('expanded_query')
                    )
                except Exception as log_error:
                    print(f"[RecallAgent.query] Warning: Failed to log query: {log_error}")
            
            return response
            
        except Exception as e:
            # Log the error
            if self.enable_logging and self.logger:
                try:
                    self.logger.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context=f"Query: {user_input}"
                    )
                except Exception as log_error:
                    print(f"[RecallAgent.query] Warning: Failed to log error: {log_error}")
            
            # Error handling
            error_msg = str(e)
            
            if 'api key' in error_msg.lower():
                return (
                    "❌ Error: Gemini API key is invalid or not set. "
                    "Please set the GEMINI_API_KEY environment variable."
                )
            elif 'rate limit' in error_msg.lower() or 'quota' in error_msg.lower():
                return (
                    "❌ Error: API rate limit exceeded. "
                    "Please wait a moment and try again."
                )
            elif 'connection' in error_msg.lower() or 'timeout' in error_msg.lower():
                return (
                    "❌ Error: Unable to connect to Gemini API. "
                    "Please check your internet connection and try again."
                )
            else:
                return (
                    f"❌ An unexpected error occurred: {error_msg}\n"
                    "Please try again or rephrase your question."
                )
    
    def get_stats(self) -> dict:
        """
        Get agent statistics
        
        Returns:
            Dictionary with agent stats
        """
        stats = {
            'total_documents': len(self.vector_store.documents),
            'top_k': self.top_k,
            'model': self.llm_client.model_name,
            'logging_enabled': self.enable_logging
        }
        
        # Add logging stats if enabled
        if self.enable_logging and self.logger:
            log_stats = self.logger.get_log_stats()
            stats.update(log_stats)
        
        return stats


if __name__ == "__main__":
    # Test the agent
    import os
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("Please set GEMINI_API_KEY environment variable")
        exit(1)
    
    # Initialize agent
    agent = RecallAgent()
    
    # Test query
    test_query = "What recalls affect 2023 Honda Civic models?"
    print(f"\nTest Query: {test_query}")
    print("=" * 60)
    
    response = agent.query(test_query)
    
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print(response)
