from typing import cast, Tuple

from transformers import HfArgumentParser

from vlac import VLACConfig, VLAC
from vlac.dataset.dataset import getDataset
from vlac.train.args import ModelArguments, DataArguments, TrainingArguments
from vlac.train.trainer import getTrainerCls


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
    dataset = getDataset(data_args.data_mixture, **dataset_args)
    train_dataset = dataset

    for param in vlac.llm.parameters():
        param.requires_grad = False
    trainer = getTrainerCls(training_args.trainer_type)(model=vlac, train_dataset=train_dataset, args=training_args, data_collator=dataset.collate_fn)
    trainer.train()


if __name__ == "__main__":
    train()
