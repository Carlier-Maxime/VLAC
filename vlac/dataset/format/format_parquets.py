import argparse
import glob
import os
from typing import Iterable
from time import time

import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq

from vlac.dataset.format.format import FormatDataset


class FormatParquetsDataset(FormatDataset):
    default_desc = 'format parquets dataset'
    
    def init_step_data(self, input_path: str, parquet_size: int) -> argparse.Namespace:
        return argparse.Namespace()

    def get_iterator(self, input_path: str, parquet_size: int) -> Iterable:
        return tqdm(glob.glob(os.path.join(input_path, "**/*.parquet"), recursive=True), desc=self.default_desc)

    def make_df(self, data, step_data: argparse.Namespace) -> pd.DataFrame | None:
        path = data
        return pd.read_parquet(path)

    def make_last_df(self, step_data: argparse.Namespace) -> pd.DataFrame | None:
        return None

    def resume_to_samples(self, iterator: Iterable, resume_samples: int) -> Iterable:
        desc = iterator.desc if isinstance(iterator, tqdm) else self.default_desc
        iterator = iter(iterator)
        i = 0
        bar = tqdm(total=resume_samples, desc=f'resume to samples {resume_samples}', unit='samples')
        parquet_file = None
        while i < resume_samples:
            parquet_file = next(iterator)
            nb = pq.ParquetFile(parquet_file).metadata.num_rows
            bar.update(nb)
            i += nb
        files = []
        if parquet_file is not None and i > resume_samples:
            parquet_file = pq.ParquetFile(parquet_file)
            total_rows = parquet_file.metadata.num_rows
            rows_to_skip = resume_samples - i
            rows_to_keep = total_rows - rows_to_skip
            table = pq.read_table(parquet_file).slice(rows_to_skip, rows_to_keep)
            resume_parquet_file = f'/tmp/format_parquets_resume_from_{i}_at_{time()}.parquet'
            pq.write_table(table, resume_parquet_file)
            files.append(resume_parquet_file)
        files.extend(iterator)
        bar.close()
        return tqdm(files, desc=desc, unit='parquet')


if __name__ == "__main__":
    FormatParquetsDataset.format_from_args()
