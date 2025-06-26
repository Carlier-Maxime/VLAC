import json
import os
from typing import List

from safetensors.torch import safe_open


def load_weights_of_keys_start_with(path: str, start: str | List[str]):
    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    files = {}
    states = {k: {} for k in start} if isinstance(start, List) else {start: {}}
    for k, v in index.items():
        local_start = None
        for s in start:
            if not k.startswith(s):
                continue
            local_start = s
            break
        if local_start is None:
            continue
        if v not in files:
            files[v] = safe_open(os.path.join(path, v), framework="pt", device="cpu")
        f = files[v]
        key = k[len(local_start):]
        if key.startswith("."):
            key = key[1:]
        states[local_start][key] = f.get_tensor(k)
    return states
