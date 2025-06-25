import argparse
import os
import io
from argparse import Namespace
from typing import List

import torch

from vlac import VLAC
from vlac.dataset.config import COYO_PATH, EMBEDS_PATH
from vlac.dataset.dataset import getDataset, VLACDataset
from vlac.dataset.edit.editor import DatasetEditor


class MakeEmbedsDataset(DatasetEditor):
    description = "Make a Embeds Dataset for train encoder / decoder."

    def __init__(self, path: str, model: str, batch_size: int, device: str, parquet_size_mb: int):
        self.vlac = VLAC.from_pretrained(model).to(device)
        super().__init__(path)
        self.batch_size = batch_size
        self.parquet_size = parquet_size_mb << 20

    def open_dataset(self) -> VLACDataset:
        return getDataset(os.path.basename(self.path), img_preprocess=self.vlac.vision_tower.image_processor, tokenizer=self.vlac.text_tokenizer)

    def about(self, multiprocess_info: Namespace):
        print(MakeEmbedsDataset.description)

    @staticmethod
    def pre_save(tensor: torch.Tensor) -> bytes:
        data = io.BytesIO()
        torch.save(tensor.cpu().detach().clone().contiguous(), data)
        return data.getvalue()

    def split_limit(self, tensors: torch.Tensor, limits: torch.Tensor) -> List[bytes]:
        assert tensors.shape[0] == limits.shape[0]
        return [self.pre_save(tensors[i, :limits[i]]) for i in range(len(limits))]

    def _map(self, batch: dict) -> dict | None:
        device = self.vlac.device
        text_tokens, vision = batch["text_tokens"], batch["vision"].to(device)
        input_ids = text_tokens["input_ids"].to(device)
        attention_mask = text_tokens["attention_mask"].to(device)
        _, _, attention_mask, _, multimodal_embeds, multimodal_tokens = self.vlac.vlm.prepare_embeds_for_multimodal(input_ids, None, attention_mask, None, input_ids, vision, encode=False)
        ids = torch.arange(attention_mask.shape[-1], device=device).unsqueeze(0).expand(attention_mask.shape[0], -1)
        limits = torch.where(attention_mask.bool(), ids, 0).add_(1).max(dim=1)[0]
        return {
            "mask.pth": self.split_limit(attention_mask, limits),
            "embeds.pth": self.split_limit(multimodal_embeds, limits),
            "tokens.pth": self.split_limit(multimodal_tokens, limits)
        }

    def _edit(self, subdataset: VLACDataset, output_path: str) -> None:
        subdataset.map_batch(
            map_fn=self._map,
            output_path=output_path,
            batch_size=self.batch_size,
            parquet_size=self.parquet_size,
        )

    @classmethod
    def edit_from_args(cls, parser: argparse.ArgumentParser = None, input_path: str = COYO_PATH, output_path: str = EMBEDS_PATH, **_):
        if parser is None:
            parser = argparse.ArgumentParser(description=MakeEmbedsDataset.description)
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--parquet_size_mb", type=int, default=1024, help='max size in MiB occupied by the parquet once in RAM. (used for limit size by parquet in output dataset)')
        super().edit_from_args(parser, input_path, output_path, **_)


if __name__ == "__main__":
    MakeEmbedsDataset.edit_from_args()
