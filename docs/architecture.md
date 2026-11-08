# Architecture

## Overview

Rag_Langchain is an agentic RAG (Retrieval-Augmented Generation) chatbot. Users upload PDFs, which are indexed into a vector store. A deep agent then answers questions by searching chunks, delegating analysis to subagents, and synthesizing verified answers.

## Data Flow

```
User Question
    │
    ▼
┌─────────────────┐
│  Deep Agent     │◄──┐
│  (create_deep_  │   │
│   _agent)       │   │
└────────┬────────┘   │
         │            │
         ▼            │
┌─────────────────┐   │
│  Plan           │   │
│  (write_todos)  │   │
└────────┬────────┘   │
         │            │
         ▼            │
┌─────────────────┐   │
│  Search         │   │
│  (search_doc)   │   │
│  → saves chunks │   │
│  under /retrieved/││
└────────┬────────┘   │
         │            │
         ▼            │
┌─────────────────┐   │
│  Analyze        │   │
│  (task →        │   │
│   chunk-analyst │   │
│   subagent)     │───┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Synthesize     │
│  (merge summaries│
│  + cite sources)│
└────────┬────────┘
         │
         ▼
   Final Answer
```

## Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Agent | `agent.py` / `server.py` | Builds the deep agent with tools and subagents |
| PDF Reader | `filereader.py` | Extracts text from uploaded PDFs |
| Indexer | `indexing.py` | Splits text into chunks, builds vector store |
| Search Tool | `tools.py` | `search_documentation` — similarity search + saves chunks |
| Prompts | `prompts.py` | System instructions for RAG workflow + subagent delegation |
| Server | `server.py` | FastAPI bridge, SSE streaming, per-config agent caching |
| Frontend | `static/` | Chat UI, settings modal, PDF upload |

## Key Design Decisions

- **Agent cache per config**: Each `(provider, model, key, baseUrl)` tuple gets its own cached deep agent instance
- **SSE streaming**: Tokens stream from the server to the frontend with a reading-paced reveal
- **In-memory vector store**: `InMemoryVectorStore` with `HuggingFaceEmbeddings` (sentence-transformers/all-MiniLM-L6-v2)
- **Subagent delegation**: `chunk-analyst` subagent analyzes each retrieved chunk in parallel
- **Mock mode**: `MOCK_AGENT=1` serves canned responses for UI verification without API keys

## Limitations

- Vector store is in-memory; resets on server restart
- No persistence layer for uploaded PDFs beyond filesystem
- No authentication or multi-tenant isolation