from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from rag_pipeline_2.augmentation import build_augmented_messages
from rag_pipeline_2.generation import generate_response, load_model_and_tokenizer
from rag_pipeline_2.indexing import build_vector_store
from rag_pipeline_2.retriever import retrieve_relevant_documents
from rag_pipeline_2.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_store = build_vector_store()
    model, tokenizer = load_model_and_tokenizer()
    app.state.model = model
    app.state.tokenizer = tokenizer
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
def chat_completions(chat_request: ChatCompletionRequest, http_request: Request) -> ChatCompletionResponse:
    user_query = chat_request.messages[-1].content

    documents = retrieve_relevant_documents(http_request.app.state.vector_store, user_query)
    augmented_messages = build_augmented_messages(documents, user_query)
    response_text = generate_response(
        http_request.app.state.model,
        http_request.app.state.tokenizer,
        augmented_messages,
    )

    response_message = ChatMessage(role="assistant", content=response_text)
    choice = ChatCompletionResponseChoice(message=response_message)
    return ChatCompletionResponse(model=chat_request.model, choices=[choice])