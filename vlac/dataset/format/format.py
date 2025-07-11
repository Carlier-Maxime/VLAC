import argparse
import glob
import json
import os
from abc import abstractmethod, ABC
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

_pp = "_per_parquet"


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

    @staticmethod
    def assert_info(info_path: str, parquet_size: int, parquets: List[str]):
        info = json.load(open(info_path))
        assert info["parquet_count"] == len(parquets), f'the number of parquet files ({len(parquets)}) mismatch with info.json ({info["parquet_count"]}).'
        assert info[f"average_memory{_pp}"] == parquet_size, f'the average memory occupied per parquet ({info[f"average_memory{_pp}"]}) mismatch with parquet_size given ({parquet_size}).'
        assert info["total_samples"] == sum(info[f"samples{_pp}"]), f'the total number of samples ({info["total_samples"]}) mismatch with the samples per parquet {info[f"samples{_pp}"]}.'
        assert info[f"max_indices{_pp}"] == (np.array(info[f"samples{_pp}"]).cumsum() - 1).tolist(), f'the maximum indices per parquet {info[f"max_indices{_pp}"]} mismatch with samples per parquet {info[f"samples{_pp}"]}.'

    @staticmethod
    def complete_missing_info(info: dict, parquets_missing: List[str]) -> dict:
        for parquet in tqdm(parquets_missing, desc='complete missing info', unit='parquet'):
            df = pd.read_parquet(parquet)
            info[f"samples{_pp}"].append(df.shape[0])
            info[f"max_indices{_pp}"].append(info[f"max_indices{_pp}"][-1] + df.shape[0])
            info["total_samples"] += df.shape[0]
            info["parquet_count"] += 1
        return info

    def resume(self, resume_path: str, parquet_size: int) -> dict | None:
        assert os.path.isdir(resume_path), 'path must be a directory if it already exists.'
        nb_files = len(os.listdir(resume_path))
        if nb_files > 0:
            print(f'The directory already exists and not empty. The format is resumed.')
            parquets = sorted(glob.glob(os.path.join(resume_path, "*.parquet")))
            info_path = os.path.join(resume_path, "info.json")
            try:
                self.assert_info(info_path, parquet_size, parquets)
            except OSError or AssertionError as e:
                print("info.json is invalid, info backup is used to resume the format.")
                info_backup_path = os.path.join(resume_path, "info_backup.json")
                if not os.path.exists(info_backup_path): raise RuntimeError("info backup not found, and original info is invalid !", e)
                info_backup = json.load(open(info_backup_path))
                if info_backup["parquet_count"] == len(parquets):
                    os.rename(info_backup_path, info_path)
                elif info_backup["parquet_count"] < len(parquets):
                    info_backup = self.complete_missing_info(info_backup, parquets[info_backup["parquet_count"]:])
                    json.dump(info_backup, open(info_path, "w"), indent=None, separators=(',', ':'))
                    os.remove(info_backup_path)
                else:
                    raise RuntimeError(f"Resume cannot be done, info invalid and the info backup cannot be used because the number of parquet files is less than the number of parquet files in the info_backup.json file.", e)
                self.assert_info(info_path, parquet_size, parquets)
            return json.load(open(info_path))
        return None

    def format(self, input_path: str, output_path: str, parquet_size: int):
        assert os.path.abspath(input_path) != os.path.abspath(output_path), 'the input and output path must be not the same.'
        resume_info = None
        if os.path.exists(output_path):
            resume_info = self.resume(output_path, parquet_size)
        else: os.makedirs(output_path)

        part_df_mem = 0
        part_df: List[pd.DataFrame] = []
        infos_pp = {}
        i = 1
        save_paths = []
        iterator = self.get_iterator(input_path, parquet_size)
        if resume_info is not None:
            for k, v in resume_info.items():
                if k.endswith(_pp): infos_pp[k[:-len(_pp)]] = v
            save_paths = sorted(glob.glob(os.path.join(output_path, "*.parquet")))
            i += resume_info["parquet_count"]
            assert i-1 == len(save_paths)
            iterator = self.resume_to_samples(iterator, resume_info["total_samples"])
        step_data = self.init_step_data(input_path, parquet_size)
        for data in iterator:
            df = self.make_df(data, step_data)
            if df is None: continue
            part_df.append(df)
            part_df_mem += part_df[-1].memory_usage(deep=True).sum()
            while part_df_mem >= parquet_size:
                df = part_df[0] if len(part_df) == 1 else pd.concat(part_df)
                residual_df, infos = self.save_parquet(df, FormatDataset.out_path(output_path, i, save_paths), parquet_size)
                for k, v in infos.items():
                    if k not in infos_pp: infos_pp[k] = []
                    infos_pp[k].append(v)
                self.save_info(output_path, infos_pp, parquet_size)
                part_df_mem = residual_df.memory_usage(deep=True).sum()
                part_df = [residual_df]
                i += 1
        df = self.make_last_df(step_data)
        if df is not None: part_df.append(df)
        df = None if len(part_df) == 0 else part_df[0] if len(part_df) == 1 else pd.concat(part_df)
        if df is not None and df.shape[0] > 0:
            residual_df, infos = self.save_parquet(df, FormatDataset.out_path(output_path, i, save_paths), parquet_size)
            for k, v in infos.items():
                if k not in infos_pp: infos_pp[k] = []
                infos_pp[k].append(v)
            assert residual_df.shape[0] == 0
        else:
            i -= 1

        digits = len(str(len(save_paths)))
        for path in tqdm(save_paths, desc='Rename parquet files', unit='file'):
            splitext = os.path.splitext(os.path.basename(path))
            os.rename(path, os.path.join(output_path, f"{int(splitext[0]):0{digits}d}-of-{i:0{digits}d}{splitext[1]}"))

        self.save_info(output_path, infos_pp, parquet_size)

    @staticmethod
    def save_info(output_dir: str, infos_pp: dict, parquet_size: int):
        out_path = os.path.join(output_dir, "info.json")
        backup_path = os.path.join(output_dir, "info_backup.json")
        if os.path.exists(out_path):
            os.rename(out_path, backup_path)
        info = {
            "parquet_count": len(infos_pp["samples"]),
            f"average_memory{_pp}": parquet_size,
            f"max_indices{_pp}": (np.array(infos_pp["samples"]).cumsum() - 1).tolist(),
        }
        for k, v in infos_pp.items():
            info[f'{k}{_pp}'] = v
            info[f'total_{k}'] = sum(v)
        json.dump(info, open(out_path, "w"), indent=None, separators=(',', ':'))
        if os.path.exists(backup_path):
            os.remove(backup_path)

    @staticmethod
    def out_path(output: str, i: int, save_paths: List[str] = None):
        path = os.path.join(output, f"{i:09d}.parquet")
        if save_paths is not None:
            save_paths.append(path)
        return path

    def get_infos_of_parquet(self, df: pd.DataFrame) -> dict:
        return {
            "samples": df.shape[0]
        }

    def save_parquet(self, df: pd.DataFrame, output_path: str, parquet_size: int) -> Tuple[pd.DataFrame, dict]:
        mem = df.memory_usage(deep=True).sum()
        limit = int(df.shape[0] * (parquet_size / mem)) if mem > parquet_size else df.shape[0]
        save_df = df.iloc[:limit]
        infos = self.get_infos_of_parquet(save_df)
        save_df.to_parquet(output_path, engine="pyarrow", row_group_size=1, index=False)
        df = df.iloc[limit:]
        return df, infos

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

    @abstractmethod
    def resume_to_samples(self, iterator: Iterable, resume_samples: int) -> Iterable:
        raise NotImplementedError
