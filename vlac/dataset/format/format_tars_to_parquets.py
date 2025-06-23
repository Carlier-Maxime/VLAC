import argparse
import glob
from typing import Iterable

import pandas as pd
import webdataset as wds
from pympler import asizeof
from tqdm import tqdm

from vlac.dataset.format.format import FormatDataset


class FormatTarsToParquetsDataset(FormatDataset):
    def init_step_data(self, input_path: str, parquet_size: int) -> argparse.Namespace:
        return argparse.Namespace(parquet_size=parquet_size, part=[], part_mem=0)

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
        step_data.part.append(data)
        step_data.part_mem += asizeof.asizeof(data)
        if step_data.part_mem < step_data.parquet_size: return None
        df = pd.DataFrame(step_data.part)
        step_data.part = []
        return df

    def make_last_df(self, step_data: argparse.Namespace) -> pd.DataFrame | None:
        return pd.DataFrame(step_data.part) if len(step_data.part) > 0 else None


if __name__ == "__main__":
    FormatTarsToParquetsDataset.format_from_args()
