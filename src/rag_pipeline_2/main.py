import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from rag_pipeline_2.augmentation import build_augmented_messages
from rag_pipeline_2.generation import MAX_NEW_TOKENS, generate_response, stream_response
from rag_pipeline_2.indexing import build_vector_store
from rag_pipeline_2.retriever import retrieve_relevant_documents
from rag_pipeline_2.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_store = build_vector_store()
    yield


app = FastAPI(lifespan=lifespan)


def _extract_text_content(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    return " ".join(part["text"] for part in content if part.get("type") == "text")


def _format_sse_chunk(chunk: ChatCompletionChunk) -> str:
    return f"data: {chunk.model_dump_json()}\n\n"


def _stream_chat_completion_chunks(
    augmented_messages: list[ChatMessage],
    model_name: str,
    max_new_tokens: int,
) -> Iterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

    for token_text in stream_response(augmented_messages, max_new_tokens):
        chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model_name,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=token_text))],
        )
        yield _format_sse_chunk(chunk)

    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model_name,
        choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(), finish_reason="stop")],
    )
    yield _format_sse_chunk(final_chunk)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", response_model=None)
def chat_completions(
    chat_request: ChatCompletionRequest, http_request: Request
) -> ChatCompletionResponse | StreamingResponse:
    user_query = _extract_text_content(chat_request.messages[-1].content)

    documents = retrieve_relevant_documents(http_request.app.state.vector_store, user_query)
    augmented_messages = build_augmented_messages(documents, user_query)

    requested_max_tokens = chat_request.max_completion_tokens
    if requested_max_tokens is None:
        requested_max_tokens = chat_request.max_tokens
    max_new_tokens = requested_max_tokens if requested_max_tokens is not None else MAX_NEW_TOKENS

    if chat_request.stream:
        return StreamingResponse(
            _stream_chat_completion_chunks(augmented_messages, chat_request.model, max_new_tokens),
            media_type="text/event-stream",
        )

    response_text, prompt_tokens, completion_tokens = generate_response(augmented_messages, max_new_tokens)

    response_message = ChatMessage(role="assistant", content=response_text)
    choice = ChatCompletionResponseChoice(message=response_message)
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return ChatCompletionResponse(model=chat_request.model, choices=[choice], usage=usage)