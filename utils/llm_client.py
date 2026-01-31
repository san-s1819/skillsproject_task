"""
LLM Client Module for Automotive Recall Agent
Direct Gemini REST API integration (no heavy SDK dependencies)
"""

import os
import time
import json
from typing import Optional
import requests


class GeminiClient:
    """Client for Google Gemini API using direct REST calls (no SDK)"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initialize Gemini client with direct REST API
        
        Args:
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model_name = model_name
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        print(f"Initialized Gemini client with model: {model_name}")
    
    def generate_response(
        self, 
        prompt: str, 
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """
        Generate response from Gemini API with retry logic using direct REST calls
        
        Args:
            prompt: Input prompt for the model
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Prepare request payload
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }]
                }
                
                # Make API request
                response = requests.post(
                    f"{self.base_url}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30
                )
                
                # Check for errors
                if response.status_code != 200:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', response.text)
                    raise Exception(f"API error ({response.status_code}): {error_msg}")
                
                # Parse response
                data = response.json()
                
                # Extract text from response
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            return parts[0]['text']
                
                raise ValueError("Empty or invalid response from Gemini API")
                
            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                print(f"Timeout. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
                continue
                
            except requests.exceptions.ConnectionError:
                last_error = "Connection error"
                print(f"Connection error. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
                continue
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check for rate limiting
                if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                
                # For other errors, don't retry
                else:
                    raise
        
        # All retries failed
        raise Exception(f"Failed after {max_retries} attempts. Last error: {last_error}")
    
    def create_recall_prompt(self, user_query: str, retrieved_docs: list) -> str:
        """
        Create a prompt for recall queries with retrieved context and citation instructions
        
        Args:
            user_query: User's question
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            Formatted prompt string with citation support
        """
        # Build context from retrieved documents with numbered citations
        context_parts = []
        source_list = []
        
        for i, doc_result in enumerate(retrieved_docs, 1):
            doc = doc_result['document']
            filename = doc['filename']
            
            # Add numbered document with clear citation marker
            context_parts.append(f"[{i}] {filename}")
            context_parts.append(f"--- CONTENT ---")
            context_parts.append(doc['content'])
            context_parts.append("")
            
            # Build source list for reference
            source_list.append(f"[{i}] {filename}")
        
        context = "\n".join(context_parts)
        sources = "\n".join(source_list)
        
        # Create prompt with citation instructions
        prompt = f"""You are an automotive recall assistant helping customers understand vehicle recalls.

Use the following recall information to answer the user's question accurately and helpfully.

RECALL INFORMATION:
{context}

USER QUESTION:
{user_query}

INSTRUCTIONS:
- Provide a clear, helpful response based on the recall information above
- **IMPORTANT: Cite your sources using [1], [2], [3] etc. after each claim or fact**
- Include specific recall numbers (e.g., 23V-456) when relevant
- Mention affected vehicle details (year, make, model)
- Explain the issue and the remedy
- Use bullet points or structured formatting for clarity
- If the question asks about a specific vehicle and you find relevant recalls, highlight them
- If no relevant recalls are found in the provided information, say so clearly
- Be professional and safety-conscious
- **End your response with a "Sources:" section listing the documents you referenced**

RESPONSE:"""
        
        return prompt


def test_gemini_connection():
    """Test function to verify Gemini API connection"""
    try:
        client = GeminiClient()
        test_prompt = "Say 'Hello, I am working!' in one sentence."
        response = client.generate_response(test_prompt)
        print(f"✓ Gemini API connection successful!")
        print(f"Test response: {response}")
        return True
    except Exception as e:
        print(f"✗ Gemini API connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the LLM client
    test_gemini_connection()
