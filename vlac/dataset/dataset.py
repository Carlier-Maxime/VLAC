import argparse
import glob
import io
import json
import os
from collections import OrderedDict
from typing import Callable, Self, Iterable

import pandas as pd
import PIL.Image as Image
import torch.utils.data
from tqdm import tqdm

import vlac.dataset.webdataset as webdataset
from vlac.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from vlac.dataset.config import *
from vlac.dataset.format.format import FormatDataset
from vlac.dataset.format.format_dict import FormatDictDataset
from vlac.dataset.format.format_parquets import FormatParquetsDataset


class VLACDataset(torch.utils.data.Dataset):
    class PandasParquetCache:
        def __init__(self, max_loaded_files: int):
            self.max_loaded_files = max_loaded_files
            self.cache = OrderedDict()  # filepath -> pd.DataFrame

        def get(self, filepath):
            if filepath in self.cache:
                self.cache.move_to_end(filepath)
                return self.cache[filepath]

            if len(self.cache) >= self.max_loaded_files:
                self.cache.popitem(last=False)

            df = pd.read_parquet(filepath)
            self.cache[filepath] = df
            return df

        def clear(self):
            self.cache.clear()

    def __init__(self, path: str | None, keys_read: Tuple[str, ...] = None, keys_out: Tuple[str, ...] = None, cache_max_files: int = 8, **_):
        self.cache = VLACDataset.PandasParquetCache(cache_max_files)
        self.keys_read = keys_read
        self.keys_out = keys_out
        if path is None: return
        info = json.load(open(os.path.join(path, "info.json")))
        self.parquet_count = info["parquet_count"]
        self.average_memory_per_parquet = info["average_memory_per_parquet"]
        self.samples_per_parquet = info["samples_per_parquet"]
        self.max_indices_per_parquet = info["max_indices_per_parquet"]
        self.total_samples = info["total_samples"]
        self.files = sorted(glob.glob(os.path.join(path, "*.parquet")))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        i, df = self.get_df_for_sample_index(index)
        local_index = index - (1 + self.max_indices_per_parquet[i - 1] if i > 0 else 0)
        out = df.iloc[local_index]
        return out if self.keys_read is None else {self.keys_read[i] if self.keys_out is None else self.keys_out[i]: out[self.keys_read[i]] for i in range(len(self.keys_out))}

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        for i in range(self.total_samples):
            yield self[i]

    def get_df_for_sample_index(self, index):
        for i, max_i in enumerate(self.max_indices_per_parquet):
            if index <= max_i:
                return i, self.cache.get(self.files[i])
        raise ValueError(f"index {index} is probably out of range or info.json is incorrect")

    def collate_fn(self, x):
        return {k: [sample[k] for sample in x] for k in x[0].keys()}

    def getDataLoader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0, **kwargs):
        if "collate_fn" not in kwargs.keys() or kwargs["collate_fn"] is None:
            kwargs["collate_fn"] = self.collate_fn
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    def shard(self, nb_split: int, index_split: int) -> Self:
        assert 0 < nb_split <= self.parquet_count, f"nb_split must be in range [1, {self.parquet_count}]"
        assert 0 <= index_split < nb_split, f"index_split must be in range [0, {nb_split - 1}]"
        subdataset = VLACDataset(None, self.keys_read, self.keys_out, max(1, self.cache.max_loaded_files // nb_split))
        subdataset.files = self.files[index_split::nb_split]
        subdataset.parquet_count = len(subdataset.files)
        subdataset.average_memory_per_parquet = self.average_memory_per_parquet
        subdataset.samples_per_parquet = self.samples_per_parquet[index_split::nb_split]
        subdataset.max_indices_per_parquet = self.max_indices_per_parquet[index_split::nb_split]
        subdataset.total_samples = sum(subdataset.samples_per_parquet)
        return subdataset

    def __map(self, formatDataset: FormatDataset, output_path: str, load_result: bool = False, parquet_size: int = None) -> Self | None:
        parquet_size = self.average_memory_per_parquet if parquet_size is None else parquet_size
        formatDataset.format('./None', output_path, parquet_size)
        if not load_result:
            return None
        return getDataset(output_path)

    def map_parquet(self, map_fn: Callable[[pd.DataFrame], pd.DataFrame | None], output_path: str, load_result: bool = False, parquet_size: int = None) -> Self | None:
        this = self

        class FormatMapDataset(FormatParquetsDataset):
            def get_iterator(self, input_path: str, parquet_size: int) -> Iterable:
                return tqdm(this.files, desc='map dataset', unit='parquet')

            def make_df(self, data, step_data: argparse.Namespace) -> pd.DataFrame | None:
                data = super().make_df(data, step_data)
                return map_fn(data)

        return self.__map(FormatMapDataset(), output_path, load_result, parquet_size)

    def map_batch(self, map_fn: Callable[[dict], dict | None], output_path: str, batch_size: int, load_result: bool = False, parquet_size: int = None) -> Self | None:
        this = self

        class FormatMapDataset(FormatDictDataset):
            def get_iterator(self, input_path: str, parquet_size: int) -> Iterable:
                return tqdm(this.getDataLoader(batch_size=batch_size), desc='map dataset', unit='batch')

            def make_df(self, data, step_data: argparse.Namespace) -> pd.DataFrame | None:
                batch_out = map_fn(data)
                outs = [{k: batch_out[k][i] for k in batch_out.keys()} for i in range(list(batch_out.values())[0].__len__())]
                return super().make_df(outs, step_data)

        return self.__map(FormatMapDataset(), output_path, load_result, parquet_size)


class COYODataset(VLACDataset):
    def __init__(self, img_preprocess, tokenizer, **kwargs):
        super().__init__(COYO_PATH, COYO_KEYS_READ, COYO_KEYS_OUT, **kwargs)
        self.img_preprocess = img_preprocess
        self.tokenizer = tokenizer

    @staticmethod
    def __open_img(img: bytes) -> Image.Image:
        return Image.open(io.BytesIO(img)).convert("RGB")

    def __open_imgs(self, imgs: list) -> list:
        return [self.__open_img(img) if isinstance(img, bytes) else self.__open_imgs(img) for img in imgs]

    def collate_fn(self, x):
        x = super().collate_fn(x)
        img = self.img_preprocess(self.__open_imgs(x['vision']), return_tensors="pt")["pixel_values"]
        txt = x['text_tokens']
        txt = [self.__add_im_tokens(t) for t in txt] if isinstance(txt, list) else self.__add_im_tokens(txt)
        text_tokens = self.tokenizer(txt, return_tensors="pt", padding=True)
        return {
            'vision': img,
            'text_tokens': text_tokens,
        }

    @staticmethod
    def __add_im_tokens(txt):
        return f'{DEFAULT_IM_START_TOKEN}{DEFAULT_IM_END_TOKEN} : {txt}'


class COYOLabelsDataset(VLACDataset):
    def __init__(self, **kwargs):
        super().__init__(COYO_LABELS_PATH, COYO_LABELS_KEYS_READ, COYO_LABELS_KEYS_OUT, **kwargs)

    def collate_fn(self, x):
        return super().collate_fn(x)


class EmbedsDataset(VLACDataset):
    def __init__(self, **kwargs):
        super().__init__(EMBEDS_PATH, EMBEDS_KEYS_READ, EMBEDS_KEYS_OUT, **kwargs)

    def collate_fn(self, x):
        return super().collate_fn(x)


def getDataset(name: str, **kwargs) -> torch.utils.data.Dataset | VLACDataset:
    name = name.lower()
    if "webdataset" in name:
        return webdataset.getDataset(name, **kwargs)
    elif "coyo" in name:
        if "labels" in name:
            return COYOLabelsDataset(**kwargs)
        else:
            return COYODataset(**kwargs)
    elif "embeds" in name:
        return EmbedsDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
