from typing import cast, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from rag_pipeline_2.schemas import ChatMessage

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512


def load_model_and_tokenizer() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tokenizer


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[ChatMessage],
) -> str:
    chat_messages = [{"role": message.role, "content": message.content} for message in messages]
    prompt_text = cast(str, tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True))
    input_ids = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    generate = cast(Callable[..., torch.LongTensor], model.generate)
    output_ids = generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids = output_ids[0][input_ids["input_ids"].shape[1]:]
    response_text = cast(str, tokenizer.decode(generated_ids, skip_special_tokens=True))

    return response_text