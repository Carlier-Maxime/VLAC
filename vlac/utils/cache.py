from abc import ABC, abstractmethod
from collections import OrderedDict

import pandas as pd
from pyarrow import parquet as pq


class SimpleCache(ABC):
    def __init__(self, max_elements: int):
        self.max_elements = max_elements
        self.cache = OrderedDict()

    @abstractmethod
    def gen_from_key(self, key, **kwargs):
        raise KeyError

    def get(self, key, **gen_kwargs):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        if len(self.cache) >= self.max_elements:
            self.cache.popitem(last=False)

        value = self.gen_from_key(key, **gen_kwargs)
        self.cache[key] = value
        return value

    def clear(self):
        self.cache.clear()


class PandasParquetCache(SimpleCache):
    def __init__(self, max_loaded_files: int):
        super().__init__(max_loaded_files)

    def gen_from_key(self, key):
        return pd.read_parquet(key)


class PyarrowParquetFileCache(SimpleCache):
    def __init__(self, max_loaded_files: int):
        super().__init__(max_loaded_files)

    def gen_from_key(self, key, **kwargs):
        pq_file = pq.ParquetFile(key)
        assert pq_file.num_row_groups >= 10000 or pq_file.num_row_groups == pq_file.metadata.num_rows, f"parquet file {key} must be formatted with ten-thousand row group or more for performance issue with shuffle access"
        return pq_file
