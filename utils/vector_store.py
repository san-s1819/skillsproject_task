"""
Vector Store Module for Automotive Recall Agent
ChromaDB with Sentence-Transformers semantic embeddings
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np


class VectorStore:
    """Manages recall documents using ChromaDB with semantic embeddings"""
    
    def __init__(self, data_dir: str = "data/recalls", persist_dir: str = "chroma_db", 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store with ChromaDB and Sentence Transformers
        
        Args:
            data_dir: Directory containing recall documents
            persist_dir: Directory to persist ChromaDB data
            model_name: Sentence transformer model to use
        """
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.documents = []
        self.model_name = model_name
        
        # Initialize sentence transformer model
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded (embedding dimension: {self.model.get_sentence_embedding_dimension()})")
        
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
        Build ChromaDB collection with semantic embeddings
        """
        if not self.documents:
            error_msg = "No documents loaded. Call load_recall_documents() first."
            print(f"[VectorStore.build_chroma_collection] Error: {error_msg}")
            raise ValueError(error_msg)
        
        print("Building ChromaDB collection with semantic embeddings...")
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name="recall_documents")
            print("Deleted existing collection")
        except:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name="recall_documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Generate embeddings for all documents
        print("Generating embeddings...")
        texts = [doc['content'] for doc in self.documents]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(self.documents))]
        metadatas = []
        
        for doc in self.documents:
            metadata = {
                'filename': doc['filename'],
                'manufacturer': doc['metadata']['manufacturer'],
                'model': doc['metadata']['model'],
                'year': doc['metadata']['year']
            }
            metadatas.append(metadata)
        
        # Add to collection
        print("Adding documents to ChromaDB...")
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Collection built with {len(self.documents)} documents")
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> Dict:
        """
        Retrieve relevant documents using semantic search
        
        Args:
            query: User's search query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with retrieved documents, confidence, and metadata
        """
        if self.collection is None:
            error_msg = "Collection not initialized. Call initialize() first."
            print(f"[VectorStore.retrieve_documents] Error: {error_msg}")
            raise ValueError(error_msg)
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Process results
        retrieved_docs = []
        
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (cosine distance -> similarity)
                similarity_score = 1 - distance
                
                # Find original document
                doc_dict = {
                    'content': doc_content,
                    'filename': metadata['filename'],
                    'metadata': {
                        'manufacturer': metadata['manufacturer'],
                        'model': metadata['model'],
                        'year': metadata['year']
                    }
                }
                
                retrieved_docs.append({
                    'rank': i + 1,
                    'document': doc_dict,
                    'similarity_score': similarity_score,
                    'distance': distance
                })
        
        # Calculate confidence
        confidence = self.calculate_retrieval_confidence(retrieved_docs)
        
        return {
            'documents': retrieved_docs,
            'confidence': confidence,
            'query_expanded': False,  # No query expansion needed with semantic search
            'original_query': query,
            'expanded_query': query
        }
    
    def calculate_retrieval_confidence(self, retrieved_docs: List[Dict]) -> Dict:
        """
        Calculate confidence score based on similarity scores
        
        Args:
            retrieved_docs: List of retrieved documents with similarity scores
            
        Returns:
            Dictionary with confidence level, score, and reason
        """
        if not retrieved_docs:
            return {
                'level': 'LOW',
                'score': 0.0,
                'gap': 0.0,
                'reason': 'No documents retrieved'
            }
        
        top_score = retrieved_docs[0]['similarity_score']
        
        # Calculate gap between top 2 scores
        gap = 0.0
        if len(retrieved_docs) > 1:
            second_score = retrieved_docs[1]['similarity_score']
            gap = top_score - second_score
        
        # Determine confidence level (adjusted for semantic embeddings)
        if top_score > 0.7 and gap > 0.1:
            level = 'HIGH'
            reason = f'Strong semantic match (score: {top_score:.2f}, clear winner)'
        elif top_score > 0.5:
            level = 'MEDIUM'
            reason = f'Moderate semantic match (score: {top_score:.2f})'
        else:
            level = 'LOW'
            reason = f'Weak semantic match (score: {top_score:.2f})'
        
        return {
            'level': level,
            'score': top_score,
            'gap': gap,
            'reason': reason
        }
    
    def initialize(self):
        """
        Initialize the vector store: load documents and build collection
        """
        print("Initializing vector store...")
        self.load_recall_documents()
        self.build_chroma_collection()
        print("✓ Vector store initialized")


if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()
    store.initialize()
    
    # Test retrieval
    test_query = "Honda Civic brake recall"
    results = store.retrieve_documents(test_query, top_k=3)
    
    print(f"\nTest Query: {test_query}")
    print(f"Confidence: {results['confidence']['level']} - {results['confidence']['reason']}")
    print(f"\nTop {len(results['documents'])} results:")
    for doc in results['documents']:
        print(f"  [{doc['rank']}] {doc['document']['filename']} (score: {doc['similarity_score']:.4f})")
