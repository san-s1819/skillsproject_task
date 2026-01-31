"""
Vector Store Module for Automotive Recall Agent
ChromaDB with TF-IDF embeddings (lightweight, no neural models)
"""

import os
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb
from chromadb.config import Settings
import numpy as np


class VectorStore:
    """Manages recall documents using ChromaDB with TF-IDF embeddings"""
    
    def __init__(self, data_dir: str = "data/recalls", persist_dir: str = "chroma_db"):
        """
        Initialize vector store with ChromaDB and TF-IDF
        
        Args:
            data_dir: Directory containing recall documents
            persist_dir: Directory to persist ChromaDB data
        """
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.documents = []
        self.vectorizer = None
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = None
        
    def load_recall_documents(self) -> List[Dict[str, str]]:
        """
        Load all recall documents from data directory
        
        Returns:
            List of document dictionaries with content and metadata
        """
        documents = []
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.data_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract metadata from filename
                    # Format: manufacturer_model_year_component.txt
                    parts = filename.replace('.txt', '').split('_')
                    
                    doc = {
                        'content': content,
                        'filename': filename,
                        'filepath': filepath,
                        'metadata': {
                            'manufacturer': parts[0] if len(parts) > 0 else 'Unknown',
                            'model': parts[1] if len(parts) > 1 else 'Unknown',
                            'year': parts[2] if len(parts) > 2 else 'Unknown',
                        }
                    }
                    
                    documents.append(doc)
                    
                except Exception as e:
                    print(f"[VectorStore.load_recall_documents] Error loading {filename}: {type(e).__name__}: {e}")
                    continue
        
        print(f"Loaded {len(documents)} recall documents")
        self.documents = documents
        return documents
    
    def build_chroma_collection(self):
        """
        Build ChromaDB collection with TF-IDF embeddings
        """
        if not self.documents:
            error_msg = "No documents loaded. Call load_recall_documents() first."
            print(f"[VectorStore.build_chroma_collection] Error: {error_msg}")
            raise ValueError(error_msg)
        
        print("Building ChromaDB collection with TF-IDF embeddings...")
        
        # Get or create collection
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name="recall_documents")
            print(f"Loaded existing collection with {self.collection.count()} documents")
            
            # If collection exists and has correct count, we're done
            if self.collection.count() == len(self.documents):
                # Load the vectorizer metadata if available
                try:
                    metadata = self.collection.get(ids=["_vectorizer_"])
                    if metadata and metadata['documents']:
                        # Vectorizer is stored, we can use the existing collection
                        print("Using existing TF-IDF vectorizer from collection")
                        return
                except:
                    pass
                
                # Rebuild if vectorizer not found
                print("Vectorizer not found, rebuilding collection...")
                self.client.delete_collection(name="recall_documents")
                self.collection = self.client.create_collection(name="recall_documents")
            else:
                # Delete and recreate if count doesn't match
                print("Document count mismatch, rebuilding collection...")
                self.client.delete_collection(name="recall_documents")
                self.collection = self.client.create_collection(name="recall_documents")
                
        except Exception:
            # Create new collection if it doesn't exist
            self.collection = self.client.create_collection(name="recall_documents")
        
        # Build TF-IDF vectorizer
        print("Creating TF-IDF embeddings...")
        texts = [doc['content'] for doc in self.documents]
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit vocabulary size for efficiency
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        # Fit and transform documents
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Convert sparse matrix to dense for ChromaDB
        embeddings = tfidf_matrix.toarray().tolist()
        
        # Add documents to collection
        ids = []
        documents = []
        metadatas = []
        
        for i, doc in enumerate(self.documents):
            ids.append(f"doc_{i}")
            documents.append(doc['content'])
            metadatas.append({
                'filename': doc['filename'],
                'manufacturer': doc['metadata']['manufacturer'],
                'model': doc['metadata']['model'],
                'year': doc['metadata']['year']
            })
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"ChromaDB collection built with {self.collection.count()} documents")
        print(f"TF-IDF features: {len(self.vectorizer.get_feature_names_out())}")
    
    def initialize(self):
        """Initialize vector store (load or build collection)"""
        print("Initializing vector store...")
        
        # Load documents
        self.load_recall_documents()
        
        # Build or load ChromaDB collection
        self.build_chroma_collection()
        
        # Rebuild vectorizer if needed
        if self.vectorizer is None:
            print("Rebuilding TF-IDF vectorizer...")
            texts = [doc['content'] for doc in self.documents]
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.vectorizer.fit(texts)
        
        print("✓ Vector store initialized")
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with common abbreviations and synonyms for better retrieval
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query string
        """
        expanded = query
        
        # Common automotive abbreviations
        expansions = {
            'RAV4': 'RAV4 RAV-4',
            'Model 3': 'Model 3 Model3',
            'Model S': 'Model S ModelS',
            'Model X': 'Model X ModelX',
            'Model Y': 'Model Y ModelY',
            'F-150': 'F-150 F150',
            'brake': 'brake braking',
            'recall': 'recall recalls',
            'airbag': 'airbag air bag',
            'seatbelt': 'seatbelt seat belt',
        }
        
        # Apply expansions
        for abbrev, expansion in expansions.items():
            if abbrev.lower() in query.lower():
                expanded = expanded.replace(abbrev, expansion)
        
        return expanded
    
    def calculate_retrieval_confidence(self, retrieved_docs: List[Dict]) -> dict:
        """
        Calculate confidence score for retrieval quality
        
        Args:
            retrieved_docs: List of retrieved documents with similarity scores
            
        Returns:
            Dictionary with confidence level and metadata
        """
        if not retrieved_docs:
            return {
                'level': 'LOW',
                'score': 0.0,
                'reason': 'No documents retrieved'
            }
        
        scores = [doc['similarity_score'] for doc in retrieved_docs]
        top_score = scores[0]
        
        # Calculate score gap (distinctiveness)
        score_gap = scores[0] - scores[1] if len(scores) > 1 else 0
        
        # Determine confidence level
        if top_score > 0.7 and score_gap > 0.15:
            level = 'HIGH'
            reason = f'Strong match (score: {top_score:.2f}, clear winner)'
        elif top_score > 0.4:
            level = 'MEDIUM'
            reason = f'Moderate match (score: {top_score:.2f})'
        else:
            level = 'LOW'
            reason = f'Weak match (score: {top_score:.2f})'
        
        return {
            'level': level,
            'score': top_score,
            'gap': score_gap,
            'reason': reason
        }
    
    def retrieve_documents(self, query: str, top_k: int = 3, use_expansion: bool = True) -> dict:
        """
        Retrieve most relevant documents for a query with confidence scoring
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            use_expansion: Whether to apply query expansion
            
        Returns:
            Dictionary with retrieved documents, confidence, and metadata
        """
        if self.collection is None:
            error_msg = "Collection not initialized. Call initialize() first."
            print(f"[VectorStore.retrieve_documents] Error: {error_msg}")
            raise ValueError(error_msg)
        
        if self.vectorizer is None:
            error_msg = "Vectorizer not initialized. Call initialize() first."
            print(f"[VectorStore.retrieve_documents] Error: {error_msg}")
            raise ValueError(error_msg)
        
        # Apply query expansion if enabled
        original_query = query
        if use_expansion:
            query = self.expand_query(query)
            expanded = query != original_query
        else:
            expanded = False
        
        # Transform query using TF-IDF vectorizer
        query_vector = self.vectorizer.transform([query]).toarray().tolist()[0]
        
        # Search ChromaDB collection
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        # Prepare results
        retrieved_docs = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                doc_idx = int(doc_id.split('_')[1])
                
                result = {
                    'rank': i + 1,
                    'document': self.documents[doc_idx],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'distance': results['distances'][0][i]
                }
                retrieved_docs.append(result)
        
        # Calculate confidence
        confidence = self.calculate_retrieval_confidence(retrieved_docs)
        
        return {
            'documents': retrieved_docs,
            'confidence': confidence,
            'query_expanded': expanded,
            'original_query': original_query,
            'expanded_query': query if expanded else None
        }


if __name__ == "__main__":
    # Test the vector store
    vs = VectorStore()
    vs.initialize()
    
    # Test query
    test_query = "Honda Civic fuel pump recall"
    results = vs.retrieve_documents(test_query, top_k=3)
    
    print(f"\nTest Query: {test_query}")
    print(f"Retrieved {len(results)} documents:")
    for result in results:
        print(f"  Rank {result['rank']}: {result['document']['filename']} (score: {result['similarity_score']:.3f})")
