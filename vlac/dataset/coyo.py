import glob
import os

import webdataset as wds
from torch.utils.data import IterableDataset

from vlac.dataset.config import COYO_PATH, COYO_LENGTH, COYO_SHUFFLE
from vlac.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class COYOWebDatasetIterable(IterableDataset):
    def __init__(self, img_preprocess, tokenizer, tars_path_pattern=os.path.join(COYO_PATH, "tars/*.tar"), length: int = COYO_LENGTH, shuffle: int = COYO_SHUFFLE, batch_size: int = 1):
        self.length = length
        self.batch_size = batch_size
        self.img_preprocess = img_preprocess
        self.tokenizer = tokenizer
        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(sorted(glob.glob(tars_path_pattern))),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(shuffle),
            wds.decode("pil"),
            wds.to_tuple("png", "text"),
            wds.batched(batch_size),
            wds.map(self.preprocess),
        ).with_length(self.length)

    @staticmethod
    def __add_im_tokens(txt):
        return f'{DEFAULT_IM_START_TOKEN}{DEFAULT_IM_END_TOKEN} : {txt}'

    def preprocess(self, x):
        img, txt = x
        img = self.img_preprocess(img, return_tensors="pt")["pixel_values"]
        txt = [self.__add_im_tokens(t) for t in txt] if isinstance(txt, list) else self.__add_im_tokens(txt)
        text_tokens = self.tokenizer(txt, return_tensors="pt", padding=True)
        return img, text_tokens

    def __iter__(self):
        for img, txt_tokens in self.dataset:
            yield {
                "vision": img,
                "text_tokens": txt_tokens
            }

    def __len__(self):
        return self.length // self.batch_size
