from typing import cast, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import HfArgumentParser

from vlac import VLACConfig, VLAC
from vlac.dataset.coyo import COYOWebDatasetIterable
from vlac.train.args import ModelArguments, DataArguments, TrainingArguments
from vlac.train.trainer import VLACTrainer


def collate_fn(batch):
    return {
        "text_tokens": pad_sequence([x["text_tokens"] for x in batch]).permute(1, 0).contiguous(),
        "vision": torch.stack([x["vision"] for x in batch]).contiguous()
    }


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = cast(Tuple[ModelArguments, DataArguments, TrainingArguments], parser.parse_args_into_dataclasses())

    if model_args.name_or_path is not None and len(model_args.name_or_path) > 0:
        config = VLACConfig.from_pretrained(model_args.name_or_path)
        config.verbose = True
        vlac = VLAC.from_pretrained(model_args.name_or_path, config=config)
    else:
        config = VLACConfig.from_json_file(model_args.config_file)
        vlac = VLAC(config)
    vlac.to("cuda")
    dataset = COYOWebDatasetIterable(vlac.vision_tower.image_processor, vlac.text_tokenizer)
    train_dataset = dataset

    for param in vlac.llm.parameters():
        param.requires_grad = False
    trainer = VLACTrainer(model=vlac, train_dataset=train_dataset, args=training_args, data_collator=collate_fn)
    trainer.train()


if __name__ == "__main__":
    train()
