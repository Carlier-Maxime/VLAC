import abc
import argparse
import glob
import json
import os
import shutil
from abc import ABC
from argparse import Namespace
from time import time

from tqdm import tqdm
import torch.distributed as dist

from vlac.dataset.dataset import VLACDataset
from vlac.dataset.format.format import _pp
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
        digits = len(str(multiprocess_info.ntasks))
        output_path = output_dir if mono else os.path.join(output_dir, f"shard_{multiprocess_info.procid:0{digits}d}")
        self._edit(subdataset, output_path)
        if mono: return
        if multiprocess_info.barrier:
            dist.barrier()
            if not multiprocess_info.master:
                return
            if dist.is_initialized():
                dist.destroy_process_group()
            os.remove(f'{output_dir}/sync_file')
            self.concat_shards(output_dir)
        else:
            print("WARNING: No barrier has settings up, plz dont forget concat shards after all edit process finished")

    @staticmethod
    def concat_shards(output_dir: str):
        psa = 0
        ps_mismatch = False
        parquet_size = None
        nb_parquets = 0
        shards = []
        with os.scandir(output_dir) as entries:
            for entry in tqdm(entries, desc='list shards', unit='entry'):
                if not os.path.isdir(entry.path): continue
                info_path = os.path.join(entry.path, "info.json")
                if not os.path.exists(info_path): continue
                parquets = sorted(glob.glob(f'{entry.path}/*.parquet'))
                if not parquets: continue
                info = json.load(open(info_path))
                cur_ps = info['average_memory_per_parquet']
                if parquet_size is None: parquet_size = cur_ps
                if parquet_size != cur_ps:
                    ps_mismatch = True
                shards.append({
                    'path': entry.path,
                    'info': info,
                    'parquets': parquets,
                })
                nb_parquets += len(parquets)
                psa += cur_ps
        shards = sorted(shards, key=lambda x: x['path'])
        psa /= len(shards)
        if ps_mismatch:
            del shards
            print(f"Use format parquets dataset for concat shards datasets because parquet_size mismatch, parquet size used is {psa}")
            concat_dir = os.path.join(output_dir, f'../concat_result_{time()}')
            FormatParquetsDataset().format(output_dir, concat_dir, psa)
            shutil.rmtree(output_dir)
            os.rename(concat_dir, output_dir)
        else:
            i = 1
            digits = len(str(nb_parquets))
            tmp_info_path = os.path.join(output_dir, 'tmp_info.json')
            g_info = None
            for shard in tqdm(shards, desc='concat shards', unit='shard'):
                if g_info is None: g_info = shard['info']
                else:
                    for key in tqdm(g_info.keys(), desc='merge info', unit='key', leave=False):
                        if key.startswith('total_'): g_info[key] += shard['info'][key]
                        elif key.endswith(_pp):
                            if key.startswith('average_'): g_info[key] += shard['info'][key]
                            elif key.startswith('max_'):
                                max_ = max(g_info[key])
                                g_info[key].extend([e + max_ for e in shard['info'][key]])
                            else: g_info[key].extend(shard['info'][key])
                        elif key == 'parquet_count': g_info[key] += shard['info'][key]
                        else:
                            raise ValueError(f"Unknown key: {key}")
                for parquet in tqdm(shard['parquets'], desc='move parquets', unit='parquet', leave=False):
                    os.rename(parquet, os.path.join(output_dir, f"{i:0{digits}d}-of-{nb_parquets}.parquet"))
                    i += 1
                json.dump(g_info, open(tmp_info_path, "w"), indent=None, separators=(',', ':'))
            for key in tqdm(g_info.keys(), desc='finish info', unit='key'):
                if key.endswith(_pp) and key.startswith('average_'): g_info[key] /= len(shards)
            json.dump(g_info, open(os.path.join(output_dir, 'info.json'), "w"), indent=None, separators=(',', ':'))
            print("final info has been saved.")
            os.remove(tmp_info_path)
            for shard in tqdm(shards, desc='remove shards', unit='shard'):
                shutil.rmtree(shard['path'])
            print("shards has been removed.")

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
