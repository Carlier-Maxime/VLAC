from typing import cast, Tuple

from torch.nn.utils.rnn import pad_sequence
from transformers import HfArgumentParser

from vlac import VLACConfig, VLAC
from vlac.dataset.dataset import getDatasetCls
from vlac.train.args import ModelArguments, DataArguments, TrainingArguments
from vlac.train.trainer import getTrainerCls


def collate_fn(batch):
    return {key: pad_sequence([data[key][0] for data in batch], batch_first=True).contiguous() for key in batch[0].keys()}


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
    dataset_args = {
        'img_preprocess': vlac.vision_tower.image_processor,
        'tokenizer': vlac.text_tokenizer
    }
    dataset = getDatasetCls(data_args.data_mixture)(**dataset_args)
    train_dataset = dataset

    for param in vlac.llm.parameters():
        param.requires_grad = False
    trainer = getTrainerCls(training_args.trainer_type)(model=vlac, train_dataset=train_dataset, args=training_args, data_collator=collate_fn)
    trainer.train()


if __name__ == "__main__":
    train()
