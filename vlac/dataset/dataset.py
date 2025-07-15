import argparse
import glob
import io
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Callable, Self, Iterable, Any, List, Union, IO

import PIL.Image as Image
import numpy as np
import pandas as pd
from pyarrow import parquet as pq
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import BatchEncoding

import vlac.dataset.webdataset as webdataset
from vlac.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from vlac.dataset.config import *
from vlac.dataset.format.format import FormatDataset
from vlac.dataset.format.format_dict import FormatDictDataset
from vlac.dataset.format.format_parquets import FormatParquetsDataset
from vlac.utils.cache import PyarrowParquetFileCache
from vlac.video import VideoReader

IM_START_END = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IM_END_TOKEN}"


class VLACDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | None, keys_read: Tuple[str, ...] = None, keys_out: Tuple[str, ...] = None, cache_max_files: int = 8, **_):
        self.cache = PyarrowParquetFileCache(cache_max_files)
        self.keys_read = keys_read
        self.keys_out = keys_out
        if path is None: return
        info = json.load(open(os.path.join(path, "info.json")))
        for key, value in info.items():
            setattr(self, key, value)
        assert all(hasattr(self, key) for key in ["parquet_count", "samples_per_parquet", "max_indices_per_parquet", "total_samples", "average_memory_per_parquet"]), 'missing keys in info.json !'
        self.files = sorted(glob.glob(os.path.join(path, "*.parquet")))

    def __len__(self):
        return self.total_samples

    IMG_KEYS = ['img', 'picture', 'image'] + [ext.lower() for ext in Image.OPEN.keys()]
    TENSOR_KEYS = ['tensor', 'tensors', 'pth', 'safetensors']
    TENSOR_EXTENSIONS = ('.pt', '.safetensors', '.pth')
    VIDEO_KEYS = ['video', 'vid']
    VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mkv', '.mov')

    @staticmethod
    def read_jsonl(input_data: Union[str, bytes, Path, IO]) -> list[dict]:
        if hasattr(input_data, 'read'):
            lines = input_data.readlines()
        elif isinstance(input_data, (str, Path)) and Path(input_data).exists():
            with open(input_data, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        elif isinstance(input_data, str):
            lines = input_data.splitlines()
        elif isinstance(input_data, bytes):
            lines = io.StringIO(input_data.decode('utf-8')).readlines()
        else:
            raise TypeError("Unsupported input type: must be str, bytes, Path or file-like object.")
        return [json.loads(line) for line in lines if line.strip()]

    def __decode_data(self, key: str, value: Any) -> Any:
        key = key.lower()
        if isinstance(value, bytes):
            data = io.BytesIO(value)
            if key in self.IMG_KEYS or key.endswith(tuple(Image.EXTENSION.keys())): return Image.open(data).convert("RGB")
            if key in self.TENSOR_KEYS or key.endswith(self.TENSOR_EXTENSIONS): return torch.load(data)
            if key in self.VIDEO_KEYS or key.endswith(self.VIDEO_EXTENSIONS): return VideoReader(data)
            if key.endswith(".jsonl"): return self.read_jsonl(data)
        return value

    def map_keys(self, out):
        if self.keys_read is None:
            if self.keys_out is None: return {k: self.__decode_data(k, v) for k, v in out}
            return {ko: self.__decode_data(k, v) for (k, v), ko in zip(out, self.keys_out)}
        if self.keys_out is None: return {k: self.__decode_data(k, out[k]) for k in self.keys_read}
        return {ko: self.__decode_data(kr, out[kr]) for kr, ko in zip(self.keys_read, self.keys_out)}

    def read_and_map_row(self, pq_file: pq.ParquetFile, index: int):
        cum = 0
        rg_idx = None
        idx_in_rg = None
        for i in range(pq_file.num_row_groups):
            length = pq_file.metadata.row_group(i).num_rows
            if cum + length > index:
                rg_idx = i
                idx_in_rg = index - cum
                break
            cum += length
        if rg_idx is None: raise ValueError(f"{index=} is out of range")

        it = pq_file.iter_batches(batch_size=1, row_groups=[rg_idx])
        for j, batch in enumerate(it):
            if j == idx_in_rg:
                table = batch.to_table()
                return self.map_keys({col: table[col][0].as_py() for col in table.column_names})
        raise ValueError(f"{idx_in_rg=} is out of range")

    def __getitem__(self, index):
        i, pq_file = self._get_pq_file_for_sample_index(index)
        local_index = self._get_relative_index(index, i, self.max_indices_per_parquet)
        return self.read_and_map_row(pq_file, local_index)

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        for i in range(self.total_samples):
            yield self[i]

    @staticmethod
    def _get_index_of(global_index, mi_pp) -> int:
        for i, max_i in enumerate(mi_pp):
            if global_index <= max_i:
                return i
        raise ValueError(f"index {global_index} is probably out of range or info.json is incorrect")

    def _get_pq_file_for_sample_index(self, index) -> Tuple[int, pq.ParquetFile]:
        i = self._get_index_of(index, self.max_indices_per_parquet)
        return i, self.cache.get(self.files[i])

    @staticmethod
    def _get_relative_index(global_index, unit_index, max_indices_per_unit):
        return global_index - (1 + max_indices_per_unit[unit_index - 1] if unit_index > 0 else 0)

    def collate_fn(self, x):
        return {k: [sample[k] for sample in x] for k in x[0].keys()}

    def getDataLoader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0, **kwargs):
        if "collate_fn" not in kwargs.keys() or kwargs["collate_fn"] is None:
            kwargs["collate_fn"] = self.collate_fn
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    def shard(self, nb_split: int, index_split: int) -> Self:
        assert 0 < nb_split <= self.parquet_count, f"nb_split must be in range [1, {self.parquet_count}]"
        assert 0 <= index_split < nb_split, f"index_split must be in range [0, {nb_split - 1}]"
        subdataset = deepcopy(self)
        subdataset.cache.max_elements = max(1, self.cache.max_elements // nb_split)
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

            def resume_to_samples(self, iterator: Iterable, resume_samples: int) -> Iterable:
                return tqdm(torch.utils.data.DataLoader(Subset(this, range(resume_samples, len(this), 1)), batch_size=batch_size, shuffle=False, collate_fn=this.collate_fn), desc='map dataset', unit='batch')

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
        img = self.img_preprocess(x['vision'], return_tensors="pt")["pixel_values"]
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
        x = super().collate_fn(x)
        return {k: pad_sequence(v, batch_first=True).contiguous() for k, v in x.items()}


class MinerlDataset(VLACDataset):
    def __init__(self, img_preprocess, tokenizer, history_len: int, use_prompt_format: bool = False, **kwargs):
        super().__init__(MINERL_PATH, MINERL_KEYS_READ, MINERL_KEYS_OUT, **kwargs)
        assert all([hasattr(self, name) for name in [f"frames_per_parquet", "total_frames"]]), 'missing keys in info.json !'
        self.img_preprocess = img_preprocess
        self.tokenizer = tokenizer
        self.history_len = history_len
        self.length = self.total_frames - self.total_samples
        self.max_indices_per_parquet = ((np.array(self.frames_per_parquet) - np.array(self.samples_per_parquet)).cumsum() - 1).tolist()
        self.use_prompt_format = use_prompt_format

    def __len__(self):
        return self.length

    def __make_history(self, data, last_view_index):
        imgs = []
        start_hist = (last_view_index - self.history_len) + 1
        if start_hist < 0: start_hist = 0
        end_hist = last_view_index + 1
        for i in range(start_hist, end_hist):
            try:
                imgs.append(data['video'][i])
            except StopIteration:
                break
        infos = data['infos'][start_hist:end_hist - 1]
        target_info = data['infos'][end_hist - 1]
        target_img = data['video'][end_hist]
        if not self.use_prompt_format:
            return {
                'state_hist': infos,
                'view_hist': imgs,
                'target_view': target_img,
                'target_state': target_info
            }
        prompt_in = (f"Task: {'Unknown' if data['task'] is None else data['task']}\n\n"
                     f"History (from the past to the present): \n"
                     f"{'\n'.join([f'=> {IM_START_END} => {info}' for info in infos])}\n"
                     f"=> {IM_START_END} => ?\n\n")
        prompt_out = (f"Predicted View:\n{IM_START_END}\n\n"
                      f"Predicted State: {target_info}")
        imgs.append(target_img)
        return {
            'prompt_in': prompt_in,
            'prompt_out': prompt_out,
            'imgs': imgs
        }

    def __getitem__(self, index):
        i, pq_file = self._get_pq_file_for_sample_index(index)
        df_frame_index = self._get_relative_index(index, i, self.max_indices_per_parquet)
        vid_len = pq_file.read(columns=["vid_len"]).column("vid_len").to_numpy()
        max_indices_per_samples = (vid_len - 1).cumsum() - 1
        local_index = self._get_index_of(df_frame_index, max_indices_per_samples)
        data = self.read_and_map_row(pq_file, local_index)
        vid_frame_index = self._get_relative_index(df_frame_index, local_index, max_indices_per_samples)
        res = self.__make_history(data, vid_frame_index)
        res['fps'] = data['fps']
        res['task'] = data['task']
        return res

    def collate_fn(self, x):
        x = super().collate_fn(x)
        if not self.use_prompt_format: return x
        img = self.img_preprocess(x['imgs'], return_tensors="pt")["pixel_values"]
        text_tokens_in = self.tokenizer(x['prompt_in'], return_tensors="pt", padding=True)
        text_tokens_out = self.tokenizer(x['prompt_out'], return_tensors="pt", padding=True)
        text_tokens = BatchEncoding({
            k: torch.cat([text_tokens_in[k], text_tokens_out[k]], dim=1)
            for k in text_tokens_in.keys()
        })
        labels = text_tokens['input_ids'].clone()
        labels[:, :text_tokens_in['input_ids'].shape[-1]] = IGNORE_INDEX
        return {
            'vision': img,
            'text_tokens': text_tokens,
            'labels': labels
        }


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
    elif "minerl" in name:
        return MinerlDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
