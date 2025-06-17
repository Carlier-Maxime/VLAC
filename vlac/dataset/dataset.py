import glob
import os
from typing import Tuple

import torch.utils.data
import webdataset as wds
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from vlac.dataset.config import *
from vlac.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def collate_tensors(x):
    return [pad_sequence([sample[i][0] for sample in x], batch_first=True).contiguous() for i in range(len(x[0]))]


class WebDatasetIterable(IterableDataset):
    def __init__(self, tars_path_pattern, keys_read: Tuple[str, ...], keys_out: Tuple[str, ...], preprocess, length: int, shuffle: int, batch_size: int = 1, collate_fn=wds.filters.default_collation_fn, **_):
        self.length = length
        self.batch_size = batch_size
        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(sorted(glob.glob(tars_path_pattern))),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(shuffle),
            wds.decode("pil"),
            wds.to_tuple(*keys_read),
            wds.batched(batch_size, collation_fn=collate_fn),
            wds.map(preprocess),
        ).with_length(self.length)
        self.keys_out = keys_out

    def __iter__(self):
        for data in self.dataset:
            yield {k: v for k, v in zip(self.keys_out, data)}

    def __len__(self):
        return self.length // self.batch_size


class COYOWebDatasetIterable(WebDatasetIterable):
    def __init__(self,
                 img_preprocess,
                 tokenizer,
                 tars_path_pattern=os.path.join(COYO_PATH, "tars/*.tar"),
                 keys_read: Tuple[str, ...] = COYO_KEYS_READ, keys_out: Tuple[str, ...] = COYO_KEYS_OUT,
                 length: int = COYO_LENGTH,
                 shuffle: int = COYO_SHUFFLE,
                 batch_size: int = 1,
                 **_
                 ):
        super().__init__(tars_path_pattern, keys_read, keys_out, self.preprocess, length, shuffle, batch_size)
        self.img_preprocess = img_preprocess
        self.tokenizer = tokenizer

    @staticmethod
    def __add_im_tokens(txt):
        return f'{DEFAULT_IM_START_TOKEN}{DEFAULT_IM_END_TOKEN} : {txt}'

    def preprocess(self, x):
        img, txt = x
        img = self.img_preprocess(img, return_tensors="pt")["pixel_values"]
        txt = [self.__add_im_tokens(t) for t in txt] if isinstance(txt, list) else self.__add_im_tokens(txt)
        text_tokens = self.tokenizer(txt, return_tensors="pt", padding=True)
        return img, text_tokens


class EmbedsWebDatasetIterable(WebDatasetIterable):
    def __init__(self,
                 tars_path_pattern=os.path.join(EMBEDS_PATH, "*.tar"),
                 keys_read: Tuple[str, ...] = EMBEDS_KEYS_READ,
                 keys_out: Tuple[str, ...] = EMBEDS_KEYS_OUT,
                 length: int = EMBEDS_LENGTH,
                 shuffle: int = EMBEDS_SHUFFLE,
                 batch_size: int = 1,
                 **_
                 ):
        super().__init__(tars_path_pattern, keys_read, keys_out, lambda x: x, length, shuffle, batch_size, collate_tensors)


def getDatasetCls(name: str) -> type[torch.utils.data.Dataset | IterableDataset | WebDatasetIterable]:
    name = name.lower()
    if "coyo" in name:
        return COYOWebDatasetIterable
    elif "embeds" in name:
        return EmbedsWebDatasetIterable
    else:
        raise ValueError(f"Unknown dataset name: {name}")
