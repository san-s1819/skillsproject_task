"""
Automotive Recall Agent - Main Entry Point
A RAG-powered AI agent for answering automotive recall queries
"""

import os
import sys
from utils.agent import RecallAgent


def print_welcome():
    """Print welcome message"""
    print("\n" + "=" * 70)
    print("🚗  AUTOMOTIVE RECALL AGENT  🚗")
    print("=" * 70)
    print("\nWelcome! I can help you with automotive recall information.")
    print("\nExample queries:")
    print("  • What recalls affect 2023 Honda Civic models?")
    print("  • Tell me about Toyota RAV4 brake recalls")
    print("  • Are there any Tesla Model 3 recalls?")
    print("\nType 'quit' or 'exit' to end the session.")
    print("=" * 70 + "\n")


def main():
    """Main function to run the recall agent CLI"""
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("\n❌ ERROR: GEMINI_API_KEY environment variable not set!")
        print("\nPlease set your Gemini API key:")
        print("  Windows: set GEMINI_API_KEY=your_api_key_here")
        print("  Linux/Mac: export GEMINI_API_KEY=your_api_key_here")
        print("\nGet your free API key at: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    # Initialize agent
    try:
        agent = RecallAgent(data_dir="data/recalls", top_k=3)
    except FileNotFoundError as e:
        print(f"\n❌ [main] Initialization failed - Data directory not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ [main] Initialization failed - Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ [main] Initialization failed - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print welcome message
    print_welcome()
    
    # Main query loop
    while True:
        try:
            # Get user input
            user_input = input("Your question: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Automotive Recall Agent. Stay safe!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Process query
            print("\n" + "-" * 70)
            try:
                response = agent.query(user_input)
            except Exception as e:
                print(f"[main] Error processing query: {type(e).__name__}: {e}")
                raise
            print("-" * 70)
            print("\n📋 RESPONSE:\n")
            print(response)
            print("\n" + "=" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ [main] Query error - {type(e).__name__}: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
