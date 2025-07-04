import abc
import argparse
import glob
import os
import shutil
from abc import ABC
from argparse import Namespace
from time import time
from typing import override, Iterable

import torch.distributed as dist
from tqdm import tqdm

from vlac.dataset.dataset import VLACDataset
from vlac.dataset.format.format_parquets import FormatParquetsDataset
from vlac.utils.multiprocess import add_multiprocess_args, check_multiprocess_args, remove_multiprocess_args


class DatasetEditor(ABC):
    def __init__(self, path: str):
        self.path = path
        self.dataset = self.open_dataset()

    @abc.abstractmethod
    def about(self, multiprocess_info: Namespace):
        pass

    def open_dataset(self) -> VLACDataset:
        return VLACDataset(self.path)

    @abc.abstractmethod
    def _edit(self, subdataset: VLACDataset, output_path: str) -> None:
        pass

    def edit(self, output_dir, multiprocess_info: Namespace = Namespace(procid=None, ntasks=None, master=True, barrier=True)):
        if multiprocess_info.master:
            os.makedirs(output_dir, exist_ok=True)
        mono = multiprocess_info.procid is None
        if multiprocess_info.master:
            self.about(multiprocess_info)
        if not mono and multiprocess_info.barrier:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{output_dir}/sync_file",
                rank=multiprocess_info.procid,
                world_size=multiprocess_info.ntasks,
            )
        subdataset = self.dataset if mono else self.dataset.shard(multiprocess_info.ntasks, multiprocess_info.procid)
        output_path = output_dir if mono else os.path.join(output_dir, f"shard_{multiprocess_info.procid}")
        self._edit(subdataset, output_path)
        if mono: return
        if multiprocess_info.barrier:
            dist.barrier()
            if not multiprocess_info.master:
                return
            if dist.is_initialized():
                dist.destroy_process_group()
            os.remove(f'{output_dir}/sync_file')
            self.concat_shards(output_dir, self.dataset.average_memory_per_parquet)
        else:
            print("WARNING: No barrier has settings up, plz dont forget concat shards after all edit process finished")

    @staticmethod
    def concat_shards(output_dir: str, parquet_size: int = None):
        if parquet_size is None:
            parquet_size = VLACDataset(os.path.join(output_dir, 'shard_0')).average_memory_per_parquet

        concat_dir = os.path.join(output_dir, f'../concat_result_{time()}')
        FormatParquetsDataset().format(output_dir, concat_dir, parquet_size)
        shutil.rmtree(output_dir)
        os.rename(concat_dir, output_dir)

    @classmethod
    def edit_from_args(cls, parser: argparse.ArgumentParser = None, input_path: str = None, output_path: str = None, **_):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, required=input_path is None, default=input_path)
        parser.add_argument("--output", type=str, required=output_path is None, default=output_path)
        parser.add_argument("-no_barrier", dest='barrier', action='store_false', default=True)
        parser.add_argument("-finish", action='store_true', default=False)
        add_multiprocess_args(parser)
        args = parser.parse_args()
        args = check_multiprocess_args(parser, args)
        if args.finish:
            cls.concat_shards(args.output)
            return
        init_args = remove_multiprocess_args(args)
        del init_args.output
        del init_args.barrier
        del init_args.finish
        init_args.path = args.input
        del init_args.input
        cls(**vars(init_args)).edit(args.output, args)
