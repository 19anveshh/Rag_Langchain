from __future__ import annotations

import asyncio
import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from filereader import read_pdf_text
from pathlib import Path
import shutil

load_dotenv()

MOCK = os.getenv("MOCK_AGENT") == "1"
PDF_DIR = Path("pdfs")

STEP_LABELS = {
    "search_documentation": "Searching documentation",
    "write_todos": "Planning",
    "task": "Analyzing chunk (chunk-analyst)",
    "read_file": "Reading chunk",
    "ls": "Listing files",
}


def label_for(name: str) -> str:
    return STEP_LABELS.get(name, f"Running {name}")


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


if not MOCK:
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from langchain.chat_models import init_chat_model
    from prompts import (
        CHUNK_ANALYST_INSTRUCTIONS,
        RAG_WORKFLOW_INSTRUCTIONS,
        SUBAGENT_DELEGATION_INSTRUCTIONS,
    )
    from tools import backend, search_documentation

    max_concurrent_analysts = 3

    INSTRUCTIONS = (
        RAG_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
            max_concurrent_analysts=max_concurrent_analysts
        )
    )

    chunk_analyst_subagent = {
        "name": "chunk-analyst",
        "description": (
            "analyse one retreived documentation chunk file"
            "Pass the use question and a single file path under /retrieved/."
        ),
        "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
    }

    model = init_chat_model(
        model="openai/gpt-oss-120b",
        model_provider="groq",
    )

    # Rebuilt here instead of importing agent.py, which fires an example
    # query at module import time.
    agent = create_deep_agent(
        model=model,
        tools=[search_documentation],
        backend=backend,
        system_prompt=INSTRUCTIONS,
        subagents=[chunk_analyst_subagent],
    )


_agent_cache: dict = {}


def get_agent(cfg: AgentConfig | None):
    if cfg is None or not (cfg.apiKey or cfg.model or cfg.baseUrl or cfg.provider):
        return agent
    provider = cfg.provider or DEFAULT_PROVIDER
    if provider not in PROVIDER_ENV:
        raise ValueError(f"Unknown provider: {provider}")
    cache_key = (provider, cfg.model or DEFAULT_MODEL, cfg.apiKey, cfg.baseUrl)
    if cache_key not in _agent_cache:
        from deepagents import create_deep_agent

        _agent_cache[cache_key] = create_deep_agent(
            model=build_model(cfg),
            tools=[search_documentation],
            backend=backend,
            system_prompt=INSTRUCTIONS,
            subagents=[chunk_analyst_subagent],
        )
    return _agent_cache[cache_key]


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


class ChatMessage(BaseModel):
    role: str
    content: str


class AgentConfig(BaseModel):
    provider: str | None = None
    model: str | None = None
    apiKey: str | None = None
    baseUrl: str | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    config: AgentConfig | None = None


DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"

PROVIDER_ENV = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def build_model(cfg: AgentConfig):
    from langchain.chat_models import init_chat_model

    provider = cfg.provider or DEFAULT_PROVIDER
    model_name = cfg.model or DEFAULT_MODEL
    env_name = PROVIDER_ENV[provider]
    if cfg.apiKey:
        os.environ[env_name] = cfg.apiKey
    elif provider == "openai" and cfg.baseUrl:
        # Local OpenAI-compatible servers ignore the key, but the client
        # still requires one to be set.
        os.environ[env_name] = "not-needed"
    kwargs = {}
    if provider == "openai" and cfg.baseUrl:
        kwargs["base_url"] = cfg.baseUrl
    return init_chat_model(model=model_name, model_provider=provider, **kwargs)


async def run_agent(messages: list[ChatMessage], agent):
    lc_messages = [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in messages
    ]
    steps: dict[str, dict] = {}
    try:
        async for mode, payload in agent.astream(
            {"messages": lc_messages}, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                chunk, _ = payload
                if isinstance(chunk, ToolMessage):
                    continue
                tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
                if tool_call_chunks:
                    for tcc in tool_call_chunks:
                        tcc_id = getattr(tcc, "id", None)
                        tcc_name = getattr(tcc, "name", None)
                        if tcc_id and tcc_name and tcc_id not in steps:
                            step = {
                                "id": tcc_id,
                                "label": label_for(tcc_name),
                                "status": "active",
                            }
                            steps[tcc_id] = step
                            yield sse({"type": "step", **step})
                else:
                    content = text_of(getattr(chunk, "content", ""))
                    if content:
                        yield sse({"type": "token", "content": content})
            elif mode == "updates":
                if isinstance(payload, dict):
                    for node_update in payload.values():
                        if not isinstance(node_update, dict):
                            continue
                        for msg in node_update.get("messages", []):
                            if isinstance(msg, AIMessage):
                                for tc in getattr(msg, "tool_calls", None) or []:
                                    tc_id = tc.get("id")
                                    tc_name = tc.get("name")
                                    if tc_id and tc_name and tc_id not in steps:
                                        step = {
                                            "id": tc_id,
                                            "label": label_for(tc_name),
                                            "status": "active",
                                        }
                                        steps[tc_id] = step
                                        yield sse({"type": "step", **step})
                            elif (
                                isinstance(msg, ToolMessage)
                                and msg.tool_call_id in steps
                            ):
                                yield sse(
                                    {
                                        "type": "step_done",
                                        "id": msg.tool_call_id,
                                        "detail": text_of(msg.content)[:200],
                                    }
                                )
    except Exception as e:
        yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
    yield sse({"type": "done"})


MOCK_ANSWER = """## Answer from mock agent

This is a **canned response** served with `MOCK_AGENT=1` so the UI can be
verified without a Groq key.

- The agent searched the indexed PDFs and saved 4 chunks.
- A `chunk-analyst` subagent summarized each chunk.
- Results were synthesized into this answer.

```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embedding=embeddings)
hits = vector_store.similarity_search("question bank topics", k=4)
```

Swap in your `GROQ_API_KEY` (`.env`) and start the server without
`MOCK_AGENT` to get real answers from the indexed question banks."""


async def run_mock(messages: list[ChatMessage]):
    yield sse({"type": "step", "id": "m1", "label": "Searching documentation", "status": "active"})
    await asyncio.sleep(0.8)
    yield sse({"type": "step_done", "id": "m1", "detail": "saved 4 documentation chunks:\n/retrieved/a1b2/chunk_1.md\n/retrieved/a1b2/chunk_2.md"})
    yield sse({"type": "step", "id": "m2", "label": "Analyzing chunk (chunk-analyst)", "status": "active"})
    await asyncio.sleep(1.2)
    yield sse({"type": "step_done", "id": "m2", "detail": "Key API names and configuration steps extracted from chunk."})
    for i in range(0, len(MOCK_ANSWER), 3):
        yield sse({"type": "token", "content": MOCK_ANSWER[i : i + 3]})
        await asyncio.sleep(0.03)
    yield sse({"type": "done"})


app = FastAPI()


async def _error_gen(e: Exception):
    yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
    yield sse({"type": "done"})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if MOCK:
        gen = run_mock(req.messages)
        print("[mock] serving canned response", flush=True)
    else:
        try:
            agent = get_agent(req.config)
        except Exception as e:
            gen = _error_gen(e)
        else:
            gen = run_agent(req.messages, agent)
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/files")
async def list_pdfs():
    if not PDF_DIR.exists():
        return {"files": []}
    files = [p.name for p in PDF_DIR.iterdir() if p.suffix.lower() == ".pdf"]
    return {"files": sorted(files)}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are allowed"}, status_code=400)

    dest = PDF_DIR / file.filename
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        i = 1
        while dest.exists():
            dest = PDF_DIR / f"{stem}_{i}{suffix}"
            i += 1

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if not MOCK:
        try:
            text = read_pdf_text(str(dest))
            doc = Document(page_content=text, metadata={"source": str(dest)})
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
            splits = splitter.split_documents([doc])
            vector_store.add_documents(documents=splits)
            return {"ok": True, "filename": dest.name, "chunks": len(splits), "indexed": True}
        except Exception as e:
            return JSONResponse({"error": f"Saved but indexing failed: {e}"}, status_code=500)

    return {"ok": True, "filename": dest.name, "indexed": False, "note": "Mock mode — indexing skipped"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    print("RAG chat server starting on http://127.0.0.1:8000" + (" (MOCK)" if MOCK else ""))
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
