from typing import cast, Tuple

from transformers import HfArgumentParser, PretrainedConfig, AutoModelForCausalLM

from vlac import VLAC
from vlac.dataset.dataset import getDataset
from vlac.model.vlac_encode_decode import VLACEncodeDecode
from vlac.train.args import ModelArguments, DataArguments, TrainingArguments
from vlac.train.trainer import getTrainerCls


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = cast(Tuple[ModelArguments, DataArguments, TrainingArguments], parser.parse_args_into_dataclasses())

    if "EncodeDecode" in training_args.trainer_type:
        model = VLACEncodeDecode(model_args.model_name_or_path, PretrainedConfig.from_pretrained(model_args.model_name_or_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model.to("cuda")
    dataset_args = {
        'img_preprocess': model.vision_tower.image_processor,
        'tokenizer': model.text_tokenizer
    } if isinstance(model, VLAC) else {}
    if "minerl" in data_args.data_mixture.lower(): dataset_args['history_len'] = 8
    dataset = getDataset(data_args.data_mixture, **dataset_args)
    train_dataset = dataset
    resume = training_args.resume_from_checkpoint
    if resume is not None and isinstance(resume, str):
        res = resume.lower()
        if res == "none": resume = None
        elif res == "true": resume = True
        elif res == "false": resume = False
        else: res = None
        if res is not None: training_args.resume_from_checkpoint = resume

    trainer = getTrainerCls(training_args.trainer_type)(model=model, train_dataset=train_dataset, args=training_args, data_collator=dataset.collate_fn)
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    train()
