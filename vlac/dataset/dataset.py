import glob
import json
import os
from collections import OrderedDict

import pandas as pd
import torch.utils.data

import vlac.dataset.webdataset as webdataset
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

    def __init__(self, path: str, keys_read: Tuple[str, ...], keys_out: Tuple[str, ...], cache_max_files: int = 8, **_):
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
        return {self.keys_out[i]: out[self.keys_read[i]] for i in range(len(self.keys_out))}

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


def getDataset(name: str, **kwargs) -> torch.utils.data.Dataset:
    name = name.lower()
    if "webdataset" in name:
        return webdataset.getDataset(name, **kwargs)
    elif "coyo" in name:
        if "labels" in name: return VLACDataset(COYO_LABELS_PATH, COYO_LABELS_KEYS_READ, COYO_LABELS_KEYS_OUT, **kwargs)
        else: return VLACDataset(COYO_PATH, COYO_KEYS_READ, COYO_KEYS_OUT, **kwargs)
    elif "embeds" in name:
        return VLACDataset(EMBEDS_PATH, EMBEDS_KEYS_READ, EMBEDS_KEYS_OUT, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
