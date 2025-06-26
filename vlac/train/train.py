from typing import cast, Tuple

from transformers import HfArgumentParser, PretrainedConfig, PreTrainedModel

from vlac import VLAC
from vlac.dataset.dataset import getDataset
from vlac.model.vlac_encode_decode import VLACEncodeDecode
from vlac.train.args import ModelArguments, DataArguments, TrainingArguments
from vlac.train.trainer import getTrainerCls


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = cast(Tuple[ModelArguments, DataArguments, TrainingArguments], parser.parse_args_into_dataclasses())

    if "EncodeDecode" in training_args.trainer_type:
        model = VLACEncodeDecode(PretrainedConfig.from_pretrained(model_args.model_name_or_path))
    else:
        model = PreTrainedModel.from_pretrained(model_args.model_name_or_path)
    model.to("cuda")
    dataset_args = {
        'img_preprocess': model.vision_tower.image_processor,
        'tokenizer': model.text_tokenizer
    } if isinstance(model, VLAC) else {}
    dataset = getDataset(data_args.data_mixture, **dataset_args)
    train_dataset = dataset

    trainer = getTrainerCls(training_args.trainer_type)(model=model, train_dataset=train_dataset, args=training_args, data_collator=dataset.collate_fn)
    trainer.train()


if __name__ == "__main__":
    train()
