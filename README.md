# Automotive Recall Agent - Phase 1

A RAG-powered AI agent that helps customers check if their vehicle is affected by recalls using semantic search and LLM reasoning.

## 🎯 Project Overview

This agent combines:
- **ChromaDB + TF-IDF**: Lightweight keyword search with ChromaDB's clean API
- **LLM (Gemini API)**: Natural language understanding and response generation
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate, context-aware answers

**Optimized for space-constrained VMs**: Only ~100MB total install size!

## 📋 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies (Minimal):**
- `google-generativeai==0.8.3` - Gemini API client
- `chromadb==0.5.23` - Vector database (pure Python)
- `scikit-learn==1.6.1` - TF-IDF vectorization (no neural models)
- `requests==2.32.3` - HTTP library

**Total install size: ~100MB** (5x smaller than neural embedding approaches!)
**No heavy subdependencies**: No PyTorch, transformers, or CUDA

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      RecallAgent                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Retrieve relevant documents (VectorStore)        │  │
│  │  2. Construct prompt with context (LLMClient)        │  │
│  │  3. Generate response (Gemini API)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────────┐          ┌──────────────────────┐
│   VectorStore        │          │    GeminiClient      │
│                      │          │                      │
│ • Load documents     │          │ • API integration    │
│ • Build TF-IDF index │          │ • Prompt templates   │
│ • Keyword search     │          │ • Retry logic        │
│ • Cosine similarity  │          │ • Error handling     │
└──────────────────────┘          └──────────────────────┘
         │
         ▼
┌──────────────────────┐
│  15 Recall Documents │
│  (data/recalls/)     │
└──────────────────────┘
```

## 🧩 Component Details

### 1. Vector Store (`utils/vector_store.py`)

**Responsibilities:**
- Load recall documents from `data/recalls/`
- Build TF-IDF embeddings using scikit-learn (keyword-based, no neural models)
- Store in ChromaDB with automatic persistence (`chroma_db/` directory)
- Retrieve top-k relevant documents using cosine similarity

**Key Methods:**
- `initialize()`: Load or build ChromaDB collection with TF-IDF
- `retrieve_documents(query, top_k)`: Keyword similarity search

### 2. Set API Key

Get a free Gemini API key from: https://aistudio.google.com/app/apikey

**Windows:**
```bash
set GEMINI_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY=your_api_key_here
```

### 3. Run the Agent

```bash
python main.py
```

## 🚀 Usage

### Interactive CLI Mode

```bash
python main.py
```

Example session:
```
Your question: What recalls affect 2023 Honda Civic models?

🔍 Retrieving relevant recall documents...
   Retrieved 3 documents:
   - honda_civic_2023_fuel_pump.txt (relevance: 0.89)
   - honda_civic_2023_camera.txt (relevance: 0.85)
   - honda_accord_2021_transmission.txt (relevance: 0.42)

🤖 Generating response with Gemini...

📋 RESPONSE:

Based on current recall data, the 2023 Honda Civic has the following active recalls:

1. **NHTSA Recall #23V-456**: Fuel pump may fail due to manufacturing defect
   - Affected: Approximately 50,000 vehicles manufactured March-August 2023
   - Issue: Engine may stall while driving, increasing crash risk
   - Remedy: Dealers will replace fuel pump free of charge
   - Contact: Honda customer service at 1-888-234-2138

2. **NHTSA Recall #23V-789**: Rearview camera software issue
   - Issue: Intermittent display failure
   - Remedy: Dealers will update software free of charge

I recommend contacting your Honda dealer to check if your specific vehicle is affected.
```

### Programmatic Usage

```python
from utils.agent import RecallAgent

# Initialize agent
agent = RecallAgent(data_dir="data/recalls", top_k=3)

# Query
response = agent.query("Tell me about Toyota RAV4 brake recalls")
print(response)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      RecallAgent                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Retrieve relevant documents (VectorStore)        │  │
│  │  2. Construct prompt with context (LLMClient)        │  │
│  │  3. Generate response (Gemini API)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────────┐          ┌──────────────────────┐
│   VectorStore        │          │    GeminiClient      │
│                      │          │                      │
│ • Load documents     │          │ • API integration    │
│ • Generate embeddings│          │ • Prompt templates   │
│ • FAISS indexing     │          │ • Retry logic        │
│ • Semantic search    │          │ • Error handling     │
└──────────────────────┘          └──────────────────────┘
         │
         ▼
┌──────────────────────┐
│  15 Recall Documents │
│  (data/recalls/)     │
└──────────────────────┘
```

## 🧩 Component Details

### 1. Vector Store (`utils/vector_store.py`)

**Responsibilities:**
- Load recall documents from `data/recalls/`
- Generate embeddings using `all-MiniLM-L6-v2` (384-dimensional)
- Build FAISS index for fast similarity search
- Cache index to disk for faster subsequent loads
- Retrieve top-k relevant documents for queries

**Key Methods:**
- `initialize()`: Load or build vector index
- `retrieve_documents(query, top_k)`: Semantic search

### 2. LLM Client (`utils/llm_client.py`)

**Responsibilities:**
- Manage Gemini API connection
- Create prompts with retrieved context
- Generate responses with retry logic
- Handle API errors (rate limits, timeouts, network issues)

**Key Methods:**
- `generate_response(prompt)`: Call Gemini with retries
- `create_recall_prompt(query, docs)`: Format RAG prompt

### 3. Agent (`utils/agent.py`)

**Responsibilities:**
- Orchestrate RAG pipeline
- Combine retrieval + generation
- Provide user-friendly error messages

**Key Methods:**
- `query(user_input)`: Main RAG pipeline
- `get_stats()`: Agent statistics

### 4. Main CLI (`main.py`)

**Responsibilities:**
- Interactive command-line interface
- API key validation
- User input/output handling

## 🎨 Design Decisions

### Why ChromaDB + TF-IDF?
- **Best of both worlds**: ChromaDB's clean API + lightweight TF-IDF embeddings
- **Space efficient**: ~100MB total (vs 500MB+ with neural models)
- **No heavy subdependencies**: No PyTorch, transformers, or CUDA
- **Fast**: Instant TF-IDF vectorization, no GPU needed
- **Sufficient quality**: Works well for keyword-rich recall documents
- **Persistent**: ChromaDB automatically saves to disk
- **VM-friendly**: Perfect for space-constrained environments

### Why Gemini API?
- **Requirement**: Specified in problem statement
- **Quality**: Strong reasoning and instruction-following
- **Free tier**: Sufficient quota for testing
- **Latest model**: Using `gemini-2.0-flash-exp` for best performance

### Why Minimal Dependencies?
- **Space efficiency**: Only 4 packages (~100MB total)
- **No heavy ML frameworks**: No PyTorch, TensorFlow, or transformers
- **Fast installation**: Completes in seconds
- **VM-optimized**: Perfect for limited disk space
- **Clean architecture**: ChromaDB + TF-IDF is simple and effective

### Prompt Engineering Strategy

**Phase 1 Prompt Structure:**
```
1. System role: "You are an automotive recall assistant..."
2. Context: Retrieved recall documents
3. User question
4. Instructions: Format, safety awareness, clarity
```

**Key prompt features:**
- Clear role definition
- Explicit instructions for structured responses
- Safety-conscious language
- Handling of negative cases (no recalls found)

## 📊 Performance Characteristics

- **Initialization time**: ~1-2 seconds (first run), <1 second (cached)
- **Query response time**: ~2-3 seconds
  - Retrieval: ~10ms (TF-IDF is very fast)
  - LLM generation**: ~2-3 seconds
- **TF-IDF features**: 1000 (configurable)
- **Storage size**: ~100KB (TF-IDF index)
- **Memory usage**: ~50MB (minimal footprint)

## ⚠️ Known Limitations (Phase 1)

1. **No VIN Integration**: Cannot check specific VINs (requires Phase 2)
   - Test queries 2 and 5 will fail
   
2. **No Multi-turn Conversations**: Each query is independent
   - No conversation history or context retention
   
3. **No Metadata Filtering**: Cannot filter by year/make/model before retrieval
   - Relies on semantic search only
   
4. **Basic Prompting**: Simple prompt template
   - No advanced techniques (chain-of-thought, few-shot examples)
   
5. **No Caching**: Every query hits Gemini API
   - Costs API quota and adds latency

6. **No Reranking**: Uses raw FAISS similarity scores
   - May not always retrieve the most relevant documents

## 🔮 Future Improvements (Phase 2 & 3)

### Phase 2 - Advanced (4-5/5 passing)
- [ ] VIN checker tool integration
- [ ] VIN format validation
- [ ] Better prompt engineering
- [ ] Structured response formatting
- [ ] Comprehensive evaluation script

### Phase 3 - Exceptional (5/5 + extras)
- [ ] Multi-turn conversation with memory
- [ ] Metadata filtering (year, make, model)
- [ ] Hybrid search (keyword + semantic)
- [ ] Reranking for better retrieval
- [ ] Response caching
- [ ] Logging and observability
- [ ] Automated test suite
- [ ] Performance optimizations

## 🧪 Testing

### Manual Testing

Run the agent and test with queries from `test_queries.json`:

**Expected to PASS (Phase 1):**
1. ✅ "What recalls affect 2023 Honda Civic models?"
3. ✅ "Tell me about Toyota RAV4 brake recalls"
4. ✅ "Are there any Tesla Model 3 recalls?"

**Expected to FAIL (Phase 1):**
2. ❌ "Is VIN 2HGFC2F59MH123456 affected by any recalls?" (needs VIN API)
5. ❌ "Is VIN 1HGCM82633A123456 affected by recalls?" (needs VIN API)

**Target**: 3/5 queries passing

### Testing Individual Components

**Test Vector Store:**
```bash
cd utils
python vector_store.py
```

**Test LLM Client:**
```bash
cd utils
python llm_client.py
```

**Test Agent:**
```bash
cd utils
python agent.py
```

## 📁 Project Structure

```
recall_agent_assessment/
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── utils/
│   ├── __init__.py                 # Package initialization
│   ├── vector_store.py             # TF-IDF vector search
│   ├── llm_client.py               # Gemini API client
│   └── agent.py                    # RAG orchestration
├── data/
│   └── recalls/                    # 15 recall documents
│       ├── honda_civic_2023_fuel_pump.txt
│       ├── honda_civic_2023_camera.txt
│       └── ... (13 more files)
├── test_queries.json               # Test cases
├── mock_vin_api.py                 # Mock VIN API server
└── chroma_db/                      # ChromaDB storage (generated, ~2MB)
```

## 🐛 Troubleshooting

### "GEMINI_API_KEY environment variable not set"
- Set the API key as shown in Setup Instructions
- Verify with: `echo %GEMINI_API_KEY%` (Windows) or `echo $GEMINI_API_KEY` (Linux/Mac)

### "Failed to load index"
- Delete `chroma_db/` directory and restart (will rebuild)
- Check that `data/recalls/` directory exists and contains .txt files

### "Rate limit exceeded"
- Wait a few seconds and try again
- Gemini free tier has rate limits
- Agent has automatic retry logic with exponential backoff

### "No relevant recall information found"
- Try rephrasing your query
- Be specific about year, make, and model
- Example: "2023 Honda Civic recalls" instead of "Honda recalls"

## 📝 License

This project is for educational/assessment purposes.

## 👤 Author

Created for the RAG-Powered Automotive Recall Agent assessment.

---

**Phase 1 Status**: ✅ Complete - Ready for testing and evaluation
