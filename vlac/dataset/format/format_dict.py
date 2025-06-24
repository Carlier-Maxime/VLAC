import argparse
from abc import ABC

import pandas as pd
from pympler import asizeof

from vlac.dataset.format.format import FormatDataset


class FormatDictDataset(FormatDataset, ABC):
    def init_step_data(self, input_path: str, parquet_size: int) -> argparse.Namespace:
        return argparse.Namespace(parquet_size=parquet_size, part=[], part_mem=0)

    def make_df(self, data, step_data: argparse.Namespace) -> pd.DataFrame | None:
        step_data.part.append(data)
        step_data.part_mem += asizeof.asizeof(data)
        if step_data.part_mem < step_data.parquet_size: return None
        df = pd.DataFrame(step_data.part)
        step_data.part = []
        return df

    def make_last_df(self, step_data: argparse.Namespace) -> pd.DataFrame | None:
        return pd.DataFrame(step_data.part) if len(step_data.part) > 0 else None


if __name__ == "__main__":
    FormatDictDataset.format_from_args()
