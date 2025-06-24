import argparse
import glob
from typing import Iterable

import pandas as pd
import webdataset as wds
from tqdm import tqdm

from vlac.dataset.format.format_dict import FormatDictDataset


class FormatTarsToParquetsDataset(FormatDictDataset):
    def get_iterator(self, input_path: str, parquet_size: int) -> Iterable:
        tars = glob.glob(f'{input_path}/*.tar')
        if len(tars) == 0:
            raise ValueError(f'no tar files found in {input_path}')
        return tqdm(wds.WebDataset(tars), desc='format webdataset to parquets files', unit='samples')

    def make_df(self, data, step_data: argparse.Namespace) -> pd.DataFrame | None:
        id_ = data["__key__"]
        try:
            id_ = int(id_)
        except ValueError:
            pass
        data = {k: data[k] for k in data.keys() if not k.startswith("__")}
        data['id'] = id_
        return super().make_df(data, step_data)


if __name__ == "__main__":
    FormatTarsToParquetsDataset.format_from_args()
