import glob
from typing import cast, Tuple

import webdataset as wds
import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import HfArgumentParser

from vlac import VLACConfig, VLAC
from vlac.train.args import ModelArguments, DataArguments, TrainingArguments
from vlac.train.trainer import VLACTrainer


class COYOWebDatasetIterable(IterableDataset):
    def __init__(self, tars_path_pattern, length: int, img_preprocess, tokenizer, shuffle: int = 100):
        self.length = length
        self.img_preprocess = img_preprocess
        self.tokenizer = tokenizer
        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(sorted(glob.glob(tars_path_pattern))),
            wds.shuffle(shuffle),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode("pil"),
            wds.to_tuple("png", "text"),
            wds.map(self.preprocess),
        ).with_length(self.length)

    def preprocess(self, x):
        img, txt = x
        img = self.img_preprocess(img, return_tensors="pt")["pixel_values"][0]
        text_tokens = self.tokenizer(txt, return_tensors="pt")["input_ids"][0]
        return img, text_tokens

    def __iter__(self):
        for img, txt_tokens in self.dataset:
            yield {
                "vision": img,
                "text_tokens": txt_tokens
            }

    def __len__(self):
        return self.length


def collate_fn(batch):
    return {
        "text_tokens": pad_sequence([x["text_tokens"] for x in batch]).permute(1, 0),
        "vision": torch.stack([x["vision"] for x in batch])
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
    dataset = COYOWebDatasetIterable("/media/hdd/datasets/coyo-700m/tars/*.tar", 256, vlac.vision_tower.image_processor, vlac.text_tokenizer)
    train_dataset = dataset

    for param in vlac.llm.parameters():
        param.requires_grad = False
    trainer = VLACTrainer(model=vlac, train_dataset=train_dataset, args=training_args, data_collator=collate_fn)
    trainer.train()


if __name__ == "__main__":
    train()
