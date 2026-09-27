# LLM RAG

Local, GPU-oriented Retrieval-Augmented Generation system with hybrid retrieval, cross-encoder reranking, grounded generation, validation, FastAPI serving, and a Streamlit client.

This repository is a hands-on portfolio project. It demonstrates an end-to-end, production-oriented implementation and operational validation; it is not presented as long-term commercial production experience.

## 1. Technical scope

The implemented system includes:

- PDF ingestion and text cleaning
- configurable chunking
- dense retrieval with sentence-transformer embeddings and FAISS
- sparse retrieval with BM25
- Reciprocal Rank Fusion (RRF)
- cross-encoder reranking
- LlamaIndex integration through an adapter around the existing retrieval pipeline
- local generation with `Qwen2.5-3B-Instruct`
- source citation formatting and validation
- numeric-grounding validation
- semantic groundedness validation
- bounded correction/retry logic
- fail-closed behavior when deterministic guardrails fail
- FastAPI backend with `/health` and `/query`
- Streamlit frontend
- structured JSON request logging
- automated tests
- retrieval and generation evaluation
- SLURM execution on an NVIDIA A30 GPU node

## 2. System flow

```text
User query
   |
   v
Streamlit frontend
   |
   v
FastAPI /query
   |
   v
LlamaIndexRAGQueryEngine
   |
   v
Dense FAISS + BM25
   |
   v
RRF fusion
   |
   v
Cross-encoder reranking
   |
   v
Top-k evidence
   |
   v
Context construction
   |
   v
Qwen2.5-3B-Instruct
   |
   v
Citation + numeric + groundedness validation
   |
   +--> deterministic validation fails -> fail closed
   |
   v
Answer + validation metadata + sources
```

Default retrieval parameters:

```text
candidate_k = 25
rerank_k    = 5
rrf_k       = 60
```

## 3. Corpus

The validated demo uses a curated corpus of five research papers covering retrieval, embeddings, reranking, and RAG:

1. ColBERT
2. Dense Passage Retrieval for Open-Domain Question Answering
3. RAG for LLMs: A Survey
4. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
5. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

The application is therefore scoped to technical questions supported by this corpus; it is not a general-purpose web or enterprise search system.

## 4. Repository structure

```text
Advanced_LLM_RAG/
├── data/
├── docs/
│   └── screenshots/
├── frontend/
│   └── app.py
├── models/
├── results/
├── scripts/
├── slurm/
├── src/
│   └── llm_rag/
│       ├── api.py
│       ├── chunker.py
│       ├── citation_validator.py
│       ├── dense_retriever.py
│       ├── document_loader.py
│       ├── embedder.py
│       ├── generator.py
│       ├── groundedness_validator.py
│       ├── hybrid_retriever.py
│       ├── llamaindex_query_engine.py
│       ├── llamaindex_retriever.py
│       ├── logging_config.py
│       ├── numeric_validator.py
│       ├── rag_pipeline.py
│       ├── reranker.py
│       ├── retrieval_metrics.py
│       ├── retrieval_pipeline.py
│       ├── source_citation_formatter.py
│       ├── sparse_retriever.py
│       └── text_cleaner.py
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## 5. Validated runtime

Canonical validation environment:

- Python `3.11.11`
- NVIDIA A30 24 GB
- CUDA-enabled PyTorch `2.11.0+cu128`
- Transformers `5.16.1`
- local `Qwen2.5-3B-Instruct`
- SLURM GPU execution

`Generator` requires CUDA in the canonical cluster runtime and loads the local model in `bfloat16`.

## 6. Installation

### 6.1 Create a Python 3.11 environment

Linux example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell example:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 6.2 Install a CUDA-compatible PyTorch build

PyTorch is intentionally not pinned to a CUDA-specific wheel in `pyproject.toml`, because the correct wheel depends on the target GPU, driver, CUDA runtime, and package index.

The validated environment used:

```text
torch 2.11.0+cu128
```

Install an appropriate CUDA-enabled PyTorch build for the target environment before installing the project.

### 6.3 Install the project

```bash
pip install -e .
```

Declared dependencies include PyMuPDF, NumPy, sentence-transformers, rank-bm25, FAISS CPU, LlamaIndex Core, FastAPI, Uvicorn, pytest, httpx, Streamlit, and Transformers.

## 7. Required local artifacts

The repository intentionally excludes large or redistributable artifacts from Git:

- raw PDFs under `data/raw/`
- model weights under `models/`
- generated embeddings
- generated FAISS indexes

The generator expects the local model at:

```text
models/generation/Qwen2.5-3B-Instruct/
```

The raw document corpus and model weights must therefore be supplied before full end-to-end execution.

## 8. Data and index preparation

The repository separates preparation into explicit scripts:

```text
scripts/build_chunks.py
scripts/build_embeddings.py
scripts/build_faiss_index.py
```

Additional inspection and validation utilities are available under `scripts/`.

For SLURM execution, the repository includes corresponding job wrappers where applicable, including:

```text
slurm/build_faiss_index.sbatch
```

Because raw documents and model artifacts are deliberately not versioned, full index reconstruction starts by restoring the five source PDFs and the required local models, then running the preparation scripts in pipeline order.

## 9. Configuration

The tracked example environment file contains:

```env
RAG_API_URL=http://127.0.0.1:8765
```

Copy it when a local environment file is useful:

```bash
cp .env.example .env
```

`.env` and other local environment files are ignored by Git; `.env.example` remains tracked.

## 10. Run the backend

Activate the environment, ensure the model/index artifacts are available, then run:

```bash
uvicorn llm_rag.api:app --host 0.0.0.0 --port 8765
```

Health endpoint:

```text
GET http://127.0.0.1:8765/health
```

Swagger/OpenAPI:

```text
http://127.0.0.1:8765/docs
```

Main RAG endpoint:

```text
POST /query
```

Request body:

```json
{
  "query": "How does ColBERT compare with BERT-based rerankers in latency and computational cost?"
}
```

The response contains:

- `answer`
- validation status
- hard-guardrail status
- citation validity
- numeric-grounding validity
- groundedness status
- retrieved sources with page, chunk ID, and score

## 11. Run the frontend

In a second process:

```bash
streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

Default local address:

```text
http://127.0.0.1:8501
```

The frontend calls the backend through `RAG_API_URL`.

## 12. Remote cluster access

When Uvicorn and Streamlit run on a remote compute node, the URLs printed by those processes are not automatically reachable from a local browser.

In the validated DANTZIG/UMH workflow, services run on GPU node `g-0`, while VS Code Remote SSH is connected to the login node. The working tunnel was:

```bash
ssh -N \
  -L 8765:127.0.0.1:8765 \
  -L 8501:127.0.0.1:8501 \
  g-0
```

Then ports `8765` and `8501` are forwarded through VS Code to the local machine.

Validated interactive GPU allocation:

```bash
sinteractive --partition GPU --qos gpu -w g-0 --cpus-per-task=24 --mem=72G --gres=gpu:1
```

This cluster-specific command is an example of the environment used to validate the project, not a requirement for other platforms.


### 12.1 Portable SLURM execution


The SLURM job files are designed to be independent of any user-specific absolute path.

Jobs must be submitted from the repository root:

```bash
cd LL_scientific_RAG
sbatch slurm/<job_name>.sbatch
```

Each job uses:
```
cd "$SLURM_SUBMIT_DIR"
```
instead of a hard-coded project path. SLURM_SUBMIT_DIR is automatically defined by SLURM and points to the directory from which sbatch was executed. This allows the same job files to work after cloning the repository into a different user account or filesystem location.
When the project virtual environment is required, jobs activate it using:
```
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
```

Therefore, the expected layout is:
```
LL_scientific_RAG/
├── .venv/
├── slurm/
├── scripts/
├── src/
└── ...
```
Submit SLURM jobs from the repository root so that relative project paths and the local virtual environment resolve correctly.


## 13. Tests

Lightweight suite:

```bash
pytest -q -m "not integration" tests
```

Validated result:

```text
21 passed, 2 deselected
```

The two observed warnings were dependency deprecation warnings from the FastAPI/Starlette testing stack, not failed tests.

A real HTTP smoke test is also available:

```bash
python scripts/smoke_fastapi.py
```

It waits for `/health`, issues a real `POST /query`, and checks the structured response path.

Relevant SLURM wrappers include:

```text
slurm/smoke_fastapi.sbatch
slurm/test_rag_pipeline_gpu.sbatch
slurm/test_generator_gpu.sbatch
slurm/test_pytest_lightweight.sbatch
```

Each job uses   cd "$SLURM_SUBMIT_DIR"
instead of a hard-coded project path.
SLURM_SUBMIT_DIR is automatically defined by SLURM and points to the directory from which sbatch was executed. This allows the same job files to work after cloning the repository into a different user account or filesystem location.

When the project virtual environment is required, jobs activate it using source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

Therefore, the expected layout is

LL_scientific_RAG/
├── .venv/
├── slurm/
├── scripts/
├── src/
└── ...

Submit SLURM jobs from the repository root so that relative project paths and the local virtual environment resolve correctly.


## 14. Evaluation

### 14.1 Retrieval holdout

The frozen retrieval holdout contains 11 queries with `top_k=5`.

| System | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|
| Dense | 0.409 | 0.439 | 0.367 |
| BM25 | 0.705 | 0.742 | 0.623 |
| Hybrid RRF | 0.523 | 0.745 | 0.539 |
| Hybrid RRF + reranker | 0.568 | 0.818 | 0.602 |

The reranker improves the hybrid RRF pipeline on all three aggregate metrics. BM25 retains the highest aggregate Recall@5 and nDCG@5 on this small holdout, while the reranked hybrid system reaches the highest MRR.

Re-run on the cluster with:

```bash
sbatch slurm/evaluate_retrieval_holdout.sbatch
```

### 14.2 End-to-end generation evaluation

Six end-to-end cases were evaluated.

| Metric | Result |
|---|---:|
| Mean retrieval Recall@5 | 0.583 |
| Citation-valid rate | 0.833 |
| Numeric-grounding-valid rate | 0.833 |
| Grounded rate | 0.167 |
| Hard-guardrail pass rate | 0.833 |

Validation statuses:

```text
validated:                   1
validated_with_soft_warning: 4
rejected:                    1
```

The low strict groundedness rate is retained as a documented limitation. It reflects the behavior of the semantic groundedness validator, while deterministic citation and numeric checks can still pass independently.

Re-run with:

```bash
sbatch slurm/evaluate_generation_end_to_end.sbatch
```

Additional result files and ablations are stored under `results/`.

## 15. Guardrails and fail-closed behavior

The generation pipeline requires citation and numeric-grounding checks to pass as hard guardrails.

If deterministic checks fail after the bounded correction attempt, the answer is replaced with:

```text
The generated answer could not be validated against the retrieved evidence.
```

Semantic groundedness is reported separately. A semantically uncertain answer can therefore be returned with `validated_with_soft_warning` when the hard deterministic checks still pass.

## 16. Structured logging

The API emits JSON request logs containing:

- UTC timestamp
- log level
- logger name
- event message
- HTTP method
- request path
- response status
- request duration in milliseconds

Example event shape:

```json
{
  "timestamp": "...",
  "level": "INFO",
  "logger": "llm_rag.api",
  "message": "request_completed",
  "method": "POST",
  "path": "/query",
  "status_code": 200,
  "duration_ms": 1232.85
}
```

## 17. Known limitations

- The corpus contains only five research papers.
- Retrieval evaluation uses a small frozen holdout of 11 queries.
- End-to-end generation evaluation contains six cases.
- Strict semantic groundedness is currently the weakest measured generation metric.
- The canonical generator requires CUDA.
- Raw papers, model weights, embeddings, and indexes are not stored in Git.
- The application is not maintained as a 24/7 public service.
- Docker, MLflow, Prometheus, and Grafana are not implemented and are not claimed as part of the current system.

## 18. Technical report

A separate illustrated report documents the methodology, engineering decisions, measured results, runtime evidence, limitations, and production-scale extensions:

```text
docs/technical_report.pdf
```

The README intentionally remains focused on implementation and reproducibility; screenshots and narrative evidence belong in the report.
