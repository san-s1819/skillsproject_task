"""
Query Logger Module for Automotive Recall Agent
Logs user queries, retrieved chunks, LLM responses, and errors
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional


class QueryLogger:
    """Logs all query interactions to a text file"""
    
    def __init__(self, log_file: str = "logs/query_log.txt", use_timestamp: bool = True):
        """
        Initialize query logger
        
        Args:
            log_file: Base path to log file
            use_timestamp: If True, append timestamp to filename (creates new file per run)
        """
        # Generate timestamped filename if enabled
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.dirname(log_file)
            log_basename = os.path.basename(log_file)
            log_name, log_ext = os.path.splitext(log_basename)
            self.log_file = os.path.join(log_dir, f"{log_name}_{timestamp}{log_ext}")
        else:
            self.log_file = log_file
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create log file with header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AUTOMOTIVE RECALL AGENT - QUERY LOG\n")
            f.write(f"Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        print(f"[QueryLogger] Logging to: {self.log_file}")
    
    def log_query(
        self,
        user_query: str,
        retrieved_chunks: List[Dict],
        confidence: Dict,
        llm_response: str,
        query_expanded: bool = False,
        expanded_query: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Log a complete query interaction
        
        Args:
            user_query: User's original question
            retrieved_chunks: List of retrieved document chunks
            confidence: Confidence scoring information
            llm_response: LLM's generated response
            query_expanded: Whether query expansion was used
            expanded_query: Expanded query if applicable
            error: Error message if query failed
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            # Header
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            
            # User Query
            f.write("USER QUERY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{user_query}\n\n")
            
            # Query Expansion
            if query_expanded and expanded_query:
                f.write("QUERY EXPANSION:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Original: {user_query}\n")
                f.write(f"Expanded: {expanded_query}\n\n")
            
            # Retrieved Chunks
            f.write("RETRIEVED CHUNKS:\n")
            f.write("-" * 80 + "\n")
            if retrieved_chunks:
                for i, chunk in enumerate(retrieved_chunks, 1):
                    doc = chunk['document']
                    score = chunk['similarity_score']
                    f.write(f"\n[CHUNK {i}]\n")
                    f.write(f"Source: {doc['filename']}\n")
                    f.write(f"Similarity Score: {score:.4f}\n")
                    f.write(f"Metadata: {doc['metadata']}\n")
                    f.write(f"Content Preview: {doc['content'][:200]}...\n")
            else:
                f.write("No chunks retrieved\n")
            f.write("\n")
            
            # Confidence Score
            f.write("CONFIDENCE ASSESSMENT:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Level: {confidence.get('level', 'N/A')}\n")
            f.write(f"Score: {confidence.get('score', 0):.4f}\n")
            f.write(f"Gap: {confidence.get('gap', 0):.4f}\n")
            f.write(f"Reason: {confidence.get('reason', 'N/A')}\n\n")
            
            # Error (if any)
            if error:
                f.write("ERROR:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{error}\n\n")
            
            # LLM Response
            f.write("LLM RESPONSE:\n")
            f.write("-" * 80 + "\n")
            if llm_response:
                f.write(f"{llm_response}\n")
            else:
                f.write("No response generated\n")
            f.write("\n")
    
    def log_error(self, error_type: str, error_message: str, context: Optional[str] = None):
        """
        Log an error
        
        Args:
            error_type: Type of error (e.g., 'VectorStoreError', 'LLMError')
            error_message: Error message
            context: Additional context about the error
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "!" * 80 + "\n")
            f.write(f"ERROR LOG - {timestamp}\n")
            f.write("!" * 80 + "\n\n")
            f.write(f"Error Type: {error_type}\n")
            f.write(f"Error Message: {error_message}\n")
            if context:
                f.write(f"Context: {context}\n")
            f.write("\n")
    
    def get_log_stats(self) -> Dict:
        """
        Get statistics about logged queries
        
        Returns:
            Dictionary with log statistics
        """
        if not os.path.exists(self.log_file):
            return {'total_queries': 0, 'total_errors': 0}
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        total_queries = content.count("USER QUERY:")
        total_errors = content.count("ERROR LOG")
        
        return {
            'total_queries': total_queries,
            'total_errors': total_errors,
            'log_file': self.log_file,
            'file_size_kb': os.path.getsize(self.log_file) / 1024 if os.path.exists(self.log_file) else 0
        }


if __name__ == "__main__":
    # Test the logger
    logger = QueryLogger()
    
    # Test query log
    logger.log_query(
        user_query="What recalls affect 2023 Honda Civic?",
        retrieved_chunks=[
            {
                'document': {
                    'filename': 'honda_civic_2023_fuel_pump.txt',
                    'content': 'Test content...',
                    'metadata': {'manufacturer': 'honda', 'model': 'civic', 'year': '2023'}
                },
                'similarity_score': 0.89
            }
        ],
        confidence={'level': 'HIGH', 'score': 0.89, 'gap': 0.25, 'reason': 'Strong match'},
        llm_response="Based on the recall information, the 2023 Honda Civic has...",
        query_expanded=True,
        expanded_query="Honda Civic 2023 recalls"
    )
    
    # Test error log
    logger.log_error(
        error_type="LLMError",
        error_message="API rate limit exceeded",
        context="Retry attempt 3/3 failed"
    )
    
    # Get stats
    stats = logger.get_log_stats()
    print(f"Log stats: {stats}")
