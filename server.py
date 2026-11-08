import asyncio
import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

load_dotenv()

MOCK = os.getenv("MOCK_AGENT") == "1"

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


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


async def run_agent(messages: list[ChatMessage]):
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


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if MOCK:
        gen = run_mock(req.messages)
        print("[mock] serving canned response")
    else:
        gen = run_agent(req.messages)
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    print("RAG chat server starting on http://127.0.0.1:8000" + (" (MOCK)" if MOCK else ""))
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
