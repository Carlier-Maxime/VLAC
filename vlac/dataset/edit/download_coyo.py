import io
import re
import argparse
from typing import List, override
from argparse import Namespace

import PIL.Image as Image
import numpy as np
import pandas as pd
import requests

from vlac.dataset.config import COYO_LABELS_PATH, COYO_PATH
from vlac.dataset.dataset import VLACDataset
from vlac.dataset.edit.editor import DatasetEditor
from vlac.utils.memory_utils import format_bytes
from vlac.utils.multiprocess import parallel_apply


class DownloadCoyoDataset(DatasetEditor):
    def __init__(self, path: str, img_size_limit: int = 336, min_clip_sim: float = 0, keep_top: float = 1, forbidden_domains: List[str] = None):
        super().__init__(path)
        self.img_size_limit = img_size_limit
        self.min_clip_sim = min_clip_sim
        self.keep_top = keep_top
        self.base_columns = ["id", "url", "text"]
        self.forbidden_domains = None if forbidden_domains is None or len(forbidden_domains) == 0 else forbidden_domains
        self.forbidden_domains_pattern = None if forbidden_domains is None else re.compile(rf"\.({'|'.join(self.forbidden_domains)})(/|$)", flags=re.IGNORECASE)

    @override
    def about(self, multiprocess_info: Namespace):
        estimated_required_space_of_one_parquet = ((1 + 3 * self.img_size_limit ** 2) * (self.dataset.total_samples / self.dataset.parquet_count) * self.keep_top) / 3
        print(f"Download COYO with clip_sim>{self.min_clip_sim}, IMG_SIZE_LIMIT={self.img_size_limit}, Keep_Top {(self.keep_top * 100):.2f} %")
        print(f"Estimated required memory : {format_bytes(estimated_required_space_of_one_parquet * (1 if multiprocess_info.procid is None else multiprocess_info.ntasks))}")
        print(f"Estimated required storage : {format_bytes(self.dataset.parquet_count * estimated_required_space_of_one_parquet)}")

    @staticmethod
    def __contain_keys(df: pd.DataFrame, keys: List[str]) -> bool:
        return all([key in df.keys() for key in keys])

    def __tweak_column(self, df: pd.DataFrame, base_columns: List[str]):
        assert self.__contain_keys(df, base_columns)
        clip_bases = ["clip_sim", "clip_similarity_vitb32", "clip_similarity_vitl14"]
        if clip_bases[0] not in df.keys():
            assert self.__contain_keys(df, clip_bases[1:]), 'the dataset not contain clip information'
            df[clip_bases[0]] = np.sum([np.array(df[key]) for key in clip_bases[1:]], axis=0)
        return df

    def _download_img(self, url: str) -> bytes | None:
        try:
            if self.forbidden_domains_pattern.search(url):
                return None
            response = requests.get(url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            size_limit = self.img_size_limit
            if min(img.size) > size_limit:
                w, h = img.size
                if h < w:
                    new_h = size_limit
                    new_w = int(size_limit * w / h)
                else:
                    new_w = size_limit
                    new_h = int(size_limit * h / w)
                img = img.resize((new_w, new_h))
            if min(img.size) < 1:
                return None
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return buffered.getvalue()
        except Exception as e:
            return None

    def __map(self, df: pd.DataFrame) -> pd.DataFrame | None:
        df = self.__tweak_column(df, self.base_columns)
        df.sort_values(by="clip_sim", ascending=False, inplace=True)
        mask = np.array(df["clip_sim"]) >= self.min_clip_sim
        mask[int(df.shape[0] * self.keep_top):] = False
        if np.sum(mask) == 0: return None
        df = df.iloc[mask][self.base_columns + ["clip_sim"]]
        df['img'] = parallel_apply(df["url"], self._download_img, num_thread_per_proc=128)
        df = df[df["img"].notna()]
        return df

    @override
    def _edit(self, subdataset: VLACDataset, output_path: str) -> None:
        subdataset.map(
            self.__map,
            output_path=output_path,
            load_result=False
        )

    @classmethod
    def edit_from_args(cls, parser: argparse.ArgumentParser = None, input_path: str = COYO_LABELS_PATH, output_path: str = COYO_PATH, **_):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument("--img_size_limit", type=int, default=336)
        parser.add_argument("--min_clip_sim", type=float, default=0.6)
        parser.add_argument("--forbidden_domains", nargs='+', type=str, default=['ru', 'рф'])
        parser.add_argument("--keep_top", type=float, default=0.2)
        super().edit_from_args(parser, input_path, output_path)


if __name__ == "__main__":
    DownloadCoyoDataset.edit_from_args()
