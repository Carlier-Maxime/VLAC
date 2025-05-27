from typing import cast, Tuple

from torch.utils.data import Dataset
from torch.utils.data.dataset import _T_co
from transformers import HfArgumentParser, CLIPProcessor, CLIPImageProcessor
import PIL.Image as Image

from vlac import VLACConfig, VLAC
from vlac.train.args import ModelArguments, DataArguments, TrainingArguments
from vlac.train.trainer import VLACTrainer


dataset = {
    "img": [
        "/media/hdd/datasets/imagenet/train/n01440764/n01440764_261.JPEG",
        "/media/hdd/datasets/imagenet/train/n01917289/n01917289_220.JPEG",
        "/media/hdd/datasets/imagenet/train/n02321529/n02321529_1660.JPEG",
        "/media/hdd/datasets/imagenet/train/n02776631/n02776631_4304.JPEG"
    ],
    "text": [
        "A fish zeiofhereorahgraeiogbureamfjruioreagbuireavhuiirhvuoaergvuoraehvouihgaoiahoirrvhaoirjvioarhgioahrioghoivhoiarehviorhvoireahvaphgoi",
        "A coral zeiofhereorahgraeiogbureamfjruioreagbuireavhuiirhvuoaergvuoraehvouihgaoiahoirrvhaoirjvioarhgioahrioghoivhoiarehviorhvoireahvaphgoi",
        "A worm zeiofhereorahgraeiogbureamfjruioreagbuireavhuiirhvuoaergvuoraehvouihgaoiahoirrvhaoirjvioarhgioahrioghoivhoiarehviorhvoireahvaphgoi",
        "A sandwich zeiofhereorahgraeiogbureamfjruioreagbuireavhuiirhvuoaergvuoraehvouihgaoiahoirrvhaoirjvioarhgioahrioghoivhoiarehviorhvoireahvaphgoi"
    ]
}


class MyDataset(Dataset):
    def __init__(self, img_paths, texts):
        self.img_paths = img_paths
        self.texts = texts

    def __getitem__(self, index) -> _T_co:
        img_path = self.img_paths[index]
        text = self.texts[index]
        image = Image.open(img_path).convert("RGB")
        return {
            "vision": image,
            "prompt": text
        }

    def __len__(self):
        return len(self.img_paths)


def collate_fn(batch):
    print(batch)
    return {
        "prompt": [x["prompt"] for x in batch],
        "vision": [x["vision"] for x in batch]
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
    # train_dataset = load_dataset("/media/hdd/datasets/coyo-700m", split="train[:100]")
    train_dataset = MyDataset(dataset["img"], dataset["text"])

    for param in vlac.llm.parameters():
        param.requires_grad = False
    trainer = VLACTrainer(model=vlac, train_dataset=train_dataset, args=training_args, data_collator=collate_fn)
    trainer.train()


if __name__ == "__main__":
    train()
