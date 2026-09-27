import logging
import time

from fastapi import FastAPI, HTTPException, Request

from llm_rag.logging_config import configure_logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from llm_rag.llamaindex_query_engine import LlamaIndexRAGQueryEngine


class QueryRequest(BaseModel):
    """Structured input for a RAG query."""

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question for the RAG system.",
    )


class SourceResponse(BaseModel):
    """Source metadata returned with the generated answer."""

    citation: str
    source: str
    page: int
    chunk_id: str
    score: float | None = None


class ValidationResponse(BaseModel):
    """Guardrail and validation results for the final answer."""

    status: str
    hard_guardrails_passed: bool
    citations_valid: bool
    numeric_grounding_valid: bool
    grounded: bool


class QueryResponse(BaseModel):
    """Structured API response for a RAG query."""

    answer: str
    validation: ValidationResponse
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    """Basic service readiness response."""

    status: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the expensive RAG query engine once when the API process starts.

    Model and retrieval initialization are blocking operations, so they are
    moved to a worker thread rather than blocking the ASGI event loop.
    """
    app.state.query_engine = await run_in_threadpool(
        LlamaIndexRAGQueryEngine.from_defaults
    )

    yield

    del app.state.query_engine

configure_logging()
logger = logging.getLogger("llm_rag.api")

app = FastAPI(
    title="Advanced LLM RAG API",
    version="0.1.0",
    description=(
        "Local RAG API with hybrid retrieval, reranking, "
        "LlamaIndex integration, Qwen generation, and guardrails."
    ),
    lifespan=lifespan,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    """Return service readiness after the RAG engine has loaded."""

    return HealthResponse(
        status="ok",
    )


@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query_rag(
    request: QueryRequest,
) -> QueryResponse:
    """
    Execute one end-to-end RAG query.

    The query engine is synchronous and compute-heavy, so execution is moved
    to a worker thread to avoid blocking the FastAPI event loop.
    """

    query_engine = app.state.query_engine

    try:
        response = await run_in_threadpool(
            query_engine.query,
            request.query,
        )
    except Exception as exc:
        # Keep internal exception details out of the HTTP response.
        raise HTTPException(
            status_code=500,
            detail="RAG query failed.",
        ) from exc

    metadata = response.metadata

    sources = [
        SourceResponse(
            citation=f"[{index}]",
            source=node.node.metadata["source"],
            page=node.node.metadata["page"],
            chunk_id=node.node.metadata["chunk_id"],
            score=node.score,
        )
        for index, node in enumerate(
            response.source_nodes,
            start=1,
        )
    ]

    return QueryResponse(
        answer=str(response),
        validation=ValidationResponse(
            status=metadata["validation_status"],
            hard_guardrails_passed=metadata[
                "hard_guardrails_passed"
            ],
            citations_valid=metadata[
                "citations_valid"
            ],
            numeric_grounding_valid=metadata[
                "numeric_grounding_valid"
            ],
            grounded=metadata["grounded"],
        ),
        sources=sources,
    )
