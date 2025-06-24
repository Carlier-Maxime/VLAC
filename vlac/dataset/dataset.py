import glob
import json
import os
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch.utils.data
from torch.utils.data import Subset

import vlac.dataset.webdataset as webdataset
from vlac.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from vlac.dataset.config import *


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

    def __init__(self, path: str, keys_read: Tuple[str, ...] = None, keys_out: Tuple[str, ...] = None, cache_max_files: int = 8, **_):
        self.cache = VLACDataset.PandasParquetCache(cache_max_files)
        info = json.load(open(os.path.join(path, "info.json")))
        self.parquet_count = info["parquet_count"]
        self.average_memory_per_parquet = info["average_memory_per_parquet"]
        self.samples_per_parquet = info["samples_per_parquet"]
        self.max_indices_per_parquet = info["max_indices_per_parquet"]
        self.total_samples = info["total_samples"]
        self.files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        self.keys_read = keys_read
        self.keys_out = keys_out

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        i, df = self.get_df_for_sample_index(index)
        local_index = index - (1 + self.max_indices_per_parquet[i-1] if i > 0 else 0)
        out = df.iloc[local_index]
        return out if self.keys_read is None else {self.keys_read[i] if self.keys_out is None else self.keys_out[i]: out[self.keys_read[i]] for i in range(len(self.keys_out))}

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        for i in range(self.total_samples):
            yield self[i]

    def get_df_for_sample_index(self, index):
        for i, max_i in enumerate(self.max_indices_per_parquet):
            if index < max_i:
                return i, self.cache.get(self.files[i])
        raise ValueError(f"index {index} is probably out of range or info.json is incorrect")

    def collate_fn(self, x):
        return x

    def getDataLoader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0, **kwargs):
        if "collate_fn" not in kwargs.keys() or kwargs["collate_fn"] is None:
            kwargs["collate_fn"] = self.collate_fn
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    def shard(self, nb_split: int, index_split: int):
        ids = np.arange(self.total_samples)
        part = self.total_samples // nb_split
        return Subset(self, ids[part * index_split:part * (index_split + 1)])


class COYODataset(VLACDataset):
    def __init__(self, img_preprocess, tokenizer, **kwargs):
        super().__init__(COYO_PATH, COYO_KEYS_READ, COYO_KEYS_OUT, **kwargs)
        self.img_preprocess = img_preprocess
        self.tokenizer = tokenizer

    def collate_fn(self, x):
        img = self.img_preprocess([s['vision'] for s in x], return_tensors="pt")["pixel_values"]
        txt = [s['text_tokens'] for s in x]
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
        return x


class EmbedsDataset(VLACDataset):
    def __init__(self, **kwargs):
        super().__init__(EMBEDS_PATH, EMBEDS_KEYS_READ, EMBEDS_KEYS_OUT, **kwargs)

    def collate_fn(self, x):
        return x


def getDataset(name: str, **kwargs) -> torch.utils.data.Dataset | VLACDataset:
    name = name.lower()
    if "webdataset" in name:
        return webdataset.getDataset(name, **kwargs)
    elif "coyo" in name:
        if "labels" in name: return COYOLabelsDataset(**kwargs)
        else: return COYODataset(**kwargs)
    elif "embeds" in name:
        return EmbedsDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
