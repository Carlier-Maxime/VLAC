import re
from typing import Dict, Optional, Sequence

import torch
import transformers

from vlac.constants import DEFAULT_IM_START_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
import vlac.conversation as conversation_lib


def tokenize_conversation(
        messages: Sequence[Dict[str, str]],
        tokenizer: transformers.PreTrainedTokenizer,
        add_generation_prompt: bool = False,
        overrides: Optional[Dict[str, str]] = None,
        no_system_prompt: bool = False,
        image_generation: bool = False,
        video_generation: bool = False,
) -> torch.Tensor:
    for message in messages:
        message["value"] = message["value"].strip()

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    if no_system_prompt:
        conv.system = ""

    # Skip the first message if it is not from human
    if messages[0]["from"] != "human":
        messages = messages[1:]

    # Add a generation prompt if needed
    if add_generation_prompt:
        messages.append({"from": "gpt", "value": None})

    conv.messages = []
    for turn, message in enumerate(messages):
        role = roles[message["from"]]
        assert role == conv.roles[turn % 2]
        if overrides is not None and message["from"] in overrides:
            conv.append_message(role, overrides[message["from"]])
        else:
            conv.append_message(role, message["value"])

    prompt = conv.get_prompt()
    if image_generation:
        prompt += f" {DEFAULT_IM_START_TOKEN}"
    elif video_generation:
        prompt += f" {DEFAULT_VI_START_TOKEN}"
    else:
        pass

    return tokenizer_image_token(prompt, tokenizer, return_tensors="pt")


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = re.split(f"({DEFAULT_IMAGE_TOKEN})", prompt)
    input_ids = [tokenizer.bos_token_id]
    for chunk in prompt_chunks:
        if chunk == DEFAULT_IMAGE_TOKEN:
            input_ids.append(image_token_index)
        else:
            input_ids.extend(tokenizer(chunk).input_ids[1:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")

    return input_ids
