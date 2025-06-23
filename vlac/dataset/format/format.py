import argparse
import json
import os
from abc import abstractmethod, ABC
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


class FormatDataset(ABC):
    def __init__(self, **_):
        pass

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True, help='path to input dataset')
        parser.add_argument("--output", type=str, required=True, help='path to output dataset')
        parser.add_argument("--parquet_size", type=int, default=1024, help='max size in MiB occupied by the parquet once in RAM. (used for limit size by parquet in output dataset)')
        args = parser.parse_args()
        args.parquet_size *= 1024 ** 2
        return args

    @classmethod
    def format_from_args(cls, **kwargs):
        return cls.format_from_args_(cls.parse_args(), **kwargs)

    @classmethod
    def format_from_args_(cls, args: argparse.Namespace, **kwargs):
        return cls(**kwargs).format(args.dataset, args.output, args.parquet_size)

    def format(self, input_path: str, output_path: str, parquet_size: int):
        assert os.path.abspath(input_path) != os.path.abspath(output_path), 'the input and output path must be not the same.'
        os.makedirs(output_path, exist_ok=True)

        part_df_mem = 0
        part_df: List[pd.DataFrame] = []
        sizes = []
        i = 1
        save_paths = []
        step_data = self.init_step_data(input_path, parquet_size)
        for data in self.get_iterator(input_path, parquet_size):
            df = self.make_df(data, step_data)
            if df is None: continue
            part_df.append(df)
            part_df_mem += part_df[-1].memory_usage(deep=True).sum()
            while part_df_mem >= parquet_size:
                df = part_df[0] if len(part_df) == 1 else pd.concat(part_df)
                residual_df, saved_df_len = FormatDataset.save_parquet(df, FormatDataset.out_path(output_path, i, save_paths), parquet_size)
                sizes.append(saved_df_len)
                part_df_mem = residual_df.memory_usage(deep=True).sum()
                part_df = [residual_df]
                i += 1
        df = self.make_last_df(step_data)
        if df is not None: part_df.append(df)
        df = part_df[0] if len(part_df) == 1 else pd.concat(part_df)
        if df.shape[0] > 0:
            residual_df, saved_df_len = FormatDataset.save_parquet(df, FormatDataset.out_path(output_path, i, save_paths), parquet_size)
            assert residual_df.shape[0] == 0
            sizes.append(saved_df_len)
        else:
            i -= 1

        digits = len(str(len(save_paths)))
        i = len(sizes)
        for path in tqdm(save_paths, desc='Rename parquet files', unit='file'):
            splitext = os.path.splitext(os.path.basename(path))
            os.rename(path, os.path.join(output_path, f"{int(splitext[0]):0{digits}d}-of-{i:0{digits}d}{splitext[1]}"))

        json.dump({
            "parquet_count": len(sizes),
            "average_memory_per_parquet": parquet_size,
            "samples_per_parquet": sizes,
            "max_indices_per_parquet": (np.array(sizes).cumsum() - 1).tolist(),
            "total_samples": sum(sizes)
        }, open(os.path.join(output_path, "info.json"), "w"), indent=None, separators=(',', ':'))

    @staticmethod
    def out_path(output: str, i: int, save_paths: List[str] = None):
        path = os.path.join(output, f"{i:09d}.parquet")
        if save_paths is not None:
            save_paths.append(path)
        return path

    @staticmethod
    def save_parquet(df: pd.DataFrame, output_path: str, parquet_size: int) -> Tuple[pd.DataFrame, int]:
        mem = df.memory_usage(deep=True).sum()
        limit = int(df.shape[0] * (parquet_size / mem)) if mem > parquet_size else df.shape[0]
        save_df = df.iloc[:limit]
        save_df_len = save_df.shape[0]
        save_df.to_parquet(output_path)
        df = df.iloc[limit:]
        return df, save_df_len

    @abstractmethod
    def init_step_data(self, input_path: str, parquet_size: int) -> argparse.Namespace:
        pass

    @abstractmethod
    def get_iterator(self, input_path: str, parquet_size: int) -> Iterable:
        pass

    @abstractmethod
    def make_df(self, data, step_data: argparse.Namespace) -> pd.DataFrame | None:
        pass

    @abstractmethod
    def make_last_df(self, step_data: argparse.Namespace) -> pd.DataFrame | None:
        pass
