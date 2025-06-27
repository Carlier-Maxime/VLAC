import glob
import json
import os
from typing import List

from safetensors.torch import safe_open


def load_weights_of_keys_start_with(path: str, start: str | List[str]) -> dict:
    start_islist = isinstance(start, List)
    states = {k: {} for k in start} if start_islist else {start: {}}
    index_path = os.path.join(path, "model.safetensors.index.json")

    def get_local_start(key: str) -> str | None:
        if start_islist:
            for s in start:
                if not key.startswith(s):
                    continue
                return s
            return None
        return start if key.startswith(start) else None

    def add_tensor(f: safe_open, full_key: str, local_start: str):
        key = full_key[len(local_start):]
        if key.startswith("."):
            key = key[1:]
        states[local_start][key] = f.get_tensor(k)

    if os.path.exists(index_path):
        index = json.load(open(index_path))["weight_map"]
        files = {}
        for k, v in index.items():
            local_start = get_local_start(k)
            if local_start is None: continue
            if v not in files:
                files[v] = safe_open(os.path.join(path, v), framework="pt", device="cpu")
            f = files[v]
            add_tensor(f, k, local_start)
    else:
        if path.endswith(".safetensors"):
            f = safe_open(path, framework="pt", device="cpu")
        else:
            files = glob.glob(os.path.join(path, "*.safetensors"))
            assert len(files) == 1, f"if not index.json must be one safetensors file found in {path}."
            f = safe_open(files[0], framework="pt", device="cpu")
        for k in f.keys():
            local_start = get_local_start(k)
            if local_start is None: continue
            add_tensor(f, k, local_start)
    return states
