"""
LLM Client Module for Automotive Recall Agent
Uses Google GenAI SDK for Gemini API integration
"""

import os
from typing import List, Dict
import google.genai as genai


class GeminiClient:
    """Client for Google Gemini API using official SDK"""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initialize Gemini client with official SDK
        
        Args:
            api_key: Gemini API key (reads from GEMINI_API_KEY env var if not provided)
            model_name: Model to use (default: gemini-2.0-flash-exp)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize the GenAI client
        try:
            self.client = genai.Client(api_key=self.api_key)
            print(f"✓ Gemini client initialized (model: {self.model_name})")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {e}")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using Gemini API
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            else:
                raise ValueError("Unexpected response format from Gemini API")
                
        except Exception as e:
            error_msg = str(e)
            
            # Handle common errors
            if 'API_KEY_INVALID' in error_msg or 'invalid api key' in error_msg.lower():
                raise ValueError(
                    "Invalid Gemini API key. Please check your GEMINI_API_KEY environment variable."
                )
            elif 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                raise Exception(
                    "API rate limit exceeded. Please wait a moment and try again."
                )
            elif 'not found' in error_msg.lower() and 'model' in error_msg.lower():
                raise ValueError(
                    f"Model '{self.model_name}' not found. Please check the model name."
                )
            else:
                raise Exception(f"Gemini API error: {error_msg}")
    
    def create_recall_prompt(self, user_query: str, retrieved_docs: List[Dict]) -> str:
        """
        Create a prompt for recall-related queries with citations
        
        Args:
            user_query: User's question about recalls
            retrieved_docs: List of retrieved document chunks with metadata
            
        Returns:
            Formatted prompt string with context and instructions
        """
        # Format retrieved documents with numbered citations
        context_parts = []
        for i, doc_info in enumerate(retrieved_docs, 1):
            doc = doc_info['document']
            score = doc_info['similarity_score']
            
            context_parts.append(
                f"[{i}] {doc['filename']}\n"
                f"Relevance Score: {score:.4f}\n"
                f"Content:\n{doc['content']}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt with citation instructions
        prompt = f"""You are an expert automotive recall assistant. Answer the user's question based ONLY on the provided recall documents.

IMPORTANT CITATION RULES:
1. Use inline citations [1], [2], [3] to reference specific documents
2. Cite sources for EVERY factual claim you make
3. At the end of your response, include a "Sources:" section listing all referenced documents
4. If the documents don't contain relevant information, say so clearly

RETRIEVED RECALL DOCUMENTS:
{context}

USER QUESTION:
{user_query}

INSTRUCTIONS:
- Provide a clear, accurate answer based on the documents
- Use numbered citations [1], [2], [3] for every claim
- Be specific about recall numbers, affected vehicles, and remedies
- Include a "Sources:" section at the end listing all cited documents
- If information is missing or unclear, acknowledge this

YOUR RESPONSE:"""

        return prompt


if __name__ == "__main__":
    # Test the client
    try:
        client = GeminiClient()
        
        # Test basic generation
        test_prompt = "Explain what a vehicle recall is in one sentence."
        response = client.generate_response(test_prompt)
        print(f"\nTest Response:\n{response}")
        
        # Test recall prompt creation
        test_docs = [
            {
                'document': {
                    'filename': 'honda_civic_2023_fuel_pump.txt',
                    'content': 'Test recall content...',
                    'metadata': {'manufacturer': 'honda', 'model': 'civic', 'year': '2023'}
                },
                'similarity_score': 0.89
            }
        ]
        
        recall_prompt = client.create_recall_prompt(
            "What recalls affect Honda Civic?",
            test_docs
        )
        print(f"\nRecall Prompt Created (length: {len(recall_prompt)} chars)")
        
    except Exception as e:
        print(f"Error: {e}")
