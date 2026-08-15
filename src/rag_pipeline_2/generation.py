from collections.abc import Iterator
from threading import Thread
from typing import Callable, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)

from rag_pipeline_2.schemas import ChatMessage

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512


def load_model_and_tokenizer() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tokenizer


def _build_prompt_input_ids(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, messages: list[ChatMessage]):
    chat_messages = [{"role": message.role, "content": message.content} for message in messages]
    prompt_text = cast(str, tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True))
    return tokenizer(prompt_text, return_tensors="pt").to(model.device)


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[ChatMessage],
) -> tuple[str, int, int]:
    input_ids = _build_prompt_input_ids(tokenizer, model, messages)

    generate = cast(Callable[..., torch.LongTensor], model.generate)
    output_ids = generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids = output_ids[0][input_ids["input_ids"].shape[1]:]
    response_text = cast(str, tokenizer.decode(generated_ids, skip_special_tokens=True))

    prompt_token_count = input_ids["input_ids"].shape[1]
    completion_token_count = generated_ids.shape[0]

    return response_text, prompt_token_count, completion_token_count


def stream_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[ChatMessage],
) -> Iterator[str]:
    input_ids = _build_prompt_input_ids(tokenizer, model, messages)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generate = cast(Callable[..., None], model.generate)
    generation_kwargs = dict(input_ids, max_new_tokens=MAX_NEW_TOKENS, streamer=streamer)
    generation_thread = Thread(target=generate, kwargs=generation_kwargs)
    generation_thread.start()

    for token_text in streamer:
        yield token_text

    generation_thread.join()