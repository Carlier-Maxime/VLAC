import glob
import os

import webdataset as wds
from torch.utils.data import IterableDataset

from vlac.dataset.config import COYO_PATH, COYO_LENGTH, COYO_SHUFFLE


class COYOWebDatasetIterable(IterableDataset):
    def __init__(self, img_preprocess, tokenizer, tars_path_pattern=os.path.join(COYO_PATH, "tars/*.tar"), length: int = COYO_LENGTH, shuffle: int = COYO_SHUFFLE):
        self.length = length
        self.img_preprocess = img_preprocess
        self.tokenizer = tokenizer
        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(sorted(glob.glob(tars_path_pattern))),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(shuffle),
            wds.decode("pil"),
            wds.to_tuple("png", "text"),
            wds.map(self.preprocess),
        ).with_length(self.length)

    def preprocess(self, x):
        img, txt = x
        img = self.img_preprocess(img, return_tensors="pt")["pixel_values"][0]
        text_tokens = self.tokenizer(txt, return_tensors="pt")["input_ids"][0]
        return img, text_tokens

    def __iter__(self):
        for img, txt_tokens in self.dataset:
            yield {
                "vision": img.contiguous(),
                "text_tokens": txt_tokens.contiguous()
            }

    def __len__(self):
        return self.length
