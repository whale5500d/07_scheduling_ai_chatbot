# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Korean RAG chatbot that integrates two components:
1. **Custom Transformer** — from-scratch BPE tokenizer + Decoder-only Transformer, instruction-tuned on Korean schedule-domain ("DaySync") data.
2. **RAG Pipeline** — Document loading → chunking → embedding → vector store → retrieval → prompt augmentation → generation, backed by either Gemma 4 E2B-it (`rag_pipeline`) or LangChain (`langchain_pipeline`).

The FastAPI app exposes `POST /query` and `POST /query/stream`; the backend is selected by `RAG_BACKEND=langchain` env var (default: custom `rag_pipeline`).

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs src/ as editable package (korean-chatbot)

cp .env.example .env      # fill in GOOGLE_API_KEY, LANGSMITH_* vars
```

Required env vars (`.env`): `GOOGLE_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`

## Commands

```bash
# Run server (default: rag_pipeline backend)
uvicorn src.main:app --reload --port 8000

# Run server with LangChain backend
RAG_BACKEND=langchain uvicorn src.main:app --reload --port 8000

# All fast tests (skip heavy model-loading tests)
pytest -m "not slow"

# All tests including model-loading (requires torch/transformers/GPU)
pytest

# Single test file
pytest tests/langchain_pipeline/test_chain.py

# Run RAG pipeline standalone (no FastAPI)
python src/rag_pipeline/generator.py
```

## Architecture

```
src/
├── main.py                  # FastAPI app — lifespan indexing, /query, /query/stream, /health
├── paths.py                 # PROJECT_ROOT / SRC_DIR / DATA_DIR (single source of truth)
├── rag_pipeline/            # Custom RAG: document_loader → chunker → embedder → vector_store
│   ├── retriever.py         #   → retriever → prompt_builder → generator
│   ├── generator.py         # TextGenerator(model_name=) — swaps Gemma ↔ custom_transformer
│   └── graph_{extractor,retriever}.py  # Graph RAG extensions
└── langchain_pipeline/      # LangChain RAG: loader → splitter → embedding → vector_store → chain
    └── chain.py             # build_rag_chain() / build_answer_only_chain() via LCEL

src/custom_transformer/
└── transformer_model.py     # BPE tokenizer + Decoder-only Transformer (pure nn.Module primitives)

data/
├── daysync_manual.md        # Primary RAG source document
└── daysync_team_records.md  # Secondary RAG source document

tests/                       # Mirrors src/ structure 1:1
└── conftest.py              # Injects torch/transformers/sentence_transformers stubs
                             # (active only when real libs are absent)
```

## Key Design Decisions

**Backend switching**: `RAG_BACKEND=langchain` selects the LangChain pipeline; any other value (including unset) uses the custom `rag_pipeline`. Both share the same FastAPI endpoints — `main.py` branches in `lifespan` and per-request handlers.

**Path management**: All file paths must import from `src/paths.py` (`PROJECT_ROOT`, `SRC_DIR`, `DATA_DIR`). Never compute `Path(__file__).parent...` chains inline — these break when files move.

**Generator interface**: `TextGenerator(model_name="google/gemma-4-E2B-it")` and `TextGenerator(model_name="custom_transformer")` share the same `generate(prompt)` / `generate_stream(prompt)` contract. Custom transformer stream is fake (full generation then word-split), because token-level streaming is not yet implemented.

**Slow tests**: Tests that load real model weights are marked `@pytest.mark.slow`. Run `pytest -m "not slow"` for fast iteration. The conftest stub only activates when torch/transformers are absent — it has no effect in real environments.

**Custom transformer limits**: `CUSTOM_MAX_INPUT_TOKENS=400` truncates RAG prompts to stay within positional encoding `max_len`. Answer quality is not a goal for the custom model path — pipeline completion without errors is the acceptance criterion.
