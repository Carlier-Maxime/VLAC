import argparse
import glob
import os
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from vlac.dataset.format.format import FormatDataset


class FormatParquetsDataset(FormatDataset):
    def init_step_data(self, input_path: str, parquet_size: int) -> argparse.Namespace:
        return argparse.Namespace()

    def get_iterator(self, input_path: str, parquet_size: int) -> Iterable:
        return tqdm(glob.glob(os.path.join(input_path, "*.parquet")), desc='format parquets dataset')

    def make_df(self, data, step_data: argparse.Namespace) -> pd.DataFrame | None:
        path = data
        return pd.read_parquet(path)

    def make_last_df(self, step_data: argparse.Namespace) -> pd.DataFrame | None:
        return None


if __name__ == "__main__":
    FormatParquetsDataset.format_from_args()
