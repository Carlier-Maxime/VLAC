import argparse
import io
import re
from argparse import Namespace
from typing import List, override

import PIL.Image as Image
import pandas as pd

from vlac.dataset.config import COYO_PATH
from vlac.dataset.dataset import VLACDataset
from vlac.dataset.edit.editor import DatasetEditor
from vlac.utils.multiprocess import parallel_apply


class CheckCoyoDataset(DatasetEditor):
    def __init__(self, path: str, img_size_limit: int = 336, min_clip_sim: float = 0, forbidden_domains: List[str] = None):
        super().__init__(path)
        self.img_size_limit = img_size_limit
        self.min_clip_sim = min_clip_sim
        self.forbidden_domains = None if forbidden_domains is None or len(forbidden_domains) == 0 else forbidden_domains
        self.forbidden_domains_pattern = None if forbidden_domains is None else re.compile(rf"\.({'|'.join(self.forbidden_domains)})(/|$)", flags=re.IGNORECASE)

    @override
    def about(self, multiprocess_info: Namespace):
        print(f"Check COYO with clip_sim>{self.min_clip_sim}, IMG_SIZE_LIMIT={self.img_size_limit}, forbidden_domains={self.forbidden_domains} ")

    @override
    def preprocess_dataset(self, dataset: VLACDataset) -> VLACDataset:
        return dataset

    def _check_domain(self, url: str) -> bool:
        return True if self.forbidden_domains is None else not bool(self.forbidden_domains_pattern.search(url))

    def _check_img(self, img: bytes) -> bool:
        try:
            if img is None: return False
            img = Image.open(io.BytesIO(img)).convert("RGB")
            if min(img.size) > self.img_size_limit or min(img.size) <= 1:
                return False
            return True
        except:
            return False

    def __map(self, df: pd.DataFrame) -> pd.DataFrame | None:
        df = df[df["clip_sim"] >= self.min_clip_sim]
        df = df[df["img"].notna()]
        df = df[parallel_apply(df['url'], self._check_domain, desc='check forbidden domains')]
        df = df[parallel_apply(df["img"], self._check_img, desc='check img')]
        return df

    @override
    def _edit(self, subdataset: VLACDataset, output_path: str) -> None:
        subdataset.map(
            self.__map,
            output_path=output_path,
            load_result=False
        )

    @classmethod
    def edit_from_args(cls, parser: argparse.ArgumentParser = None, input_path: str = COYO_PATH, output_path: str = f'{COYO_PATH}/../coyo_check', **_):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument("--img_size_limit", type=int, default=336)
        parser.add_argument("--min_clip_sim", type=float, default=0.6)
        parser.add_argument("--forbidden_domains", nargs='+', type=str, default=None)
        super().edit_from_args(parser, input_path, output_path)


if __name__ == "__main__":
    CheckCoyoDataset.edit_from_args()
