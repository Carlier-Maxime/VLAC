import glob
import json
import os
import shutil
from typing import List

from safetensors.torch import safe_open, save


def __get_index_path(path: str) -> str:
    return os.path.join(path, "model.safetensors.index.json")


def load_weights(path: str, start: str | List[str] = None) -> dict:
    start_islist = False if start is None else isinstance(start, list)
    states = {k: {} for k in start} if start_islist else {} if start is None else {start: {}}
    index_path = __get_index_path(path)

    def get_local_start(key: str) -> str | None:
        if start_islist:
            for s in start:
                if not key.startswith(s):
                    continue
                return s
            return None
        elif start is None:
            return key
        return start if key.startswith(start) else None

    def add_tensor(f: safe_open, full_key: str, local_start: str):
        same = local_start == full_key
        key = full_key[0 if same else len(local_start):]
        if key.startswith("."):
            key = key[1:]
        if same:
            states[full_key] = f.get_tensor(k)
        else:
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


def weights_by_file(weights: dict, index_path: str) -> dict:
    index = json.load(open(index_path))["weight_map"]
    new_weights = {}
    for k, v in index.items():
        if v not in new_weights:
            new_weights[v] = {}
        new_weights[v][k] = weights[k]
    return new_weights


def replace_weights_in_safetensors(base_path: str, replacement_path: str, output_path: str, key_prefix: str = ""):
    assert base_path != output_path, "base path and output path must be different."
    weights = load_weights(base_path)
    replace_weights = load_weights(replacement_path)
    if len(key_prefix) > 0 and key_prefix[-1] != '.': key_prefix += '.'
    for k in replace_weights.keys():
        key = key_prefix + k
        if key in weights:
            weights[key] = replace_weights[k]
        else:
            print(f"key '{key}' not found in base model.")
    index_path = __get_index_path(base_path)

    if os.path.exists(index_path):
        assert not output_path.endswith(".safetensors"), "output path must be a directory if base path contains multiple safetensors files."
        os.makedirs(output_path, exist_ok=True)
        if output_path != base_path: shutil.copy(index_path, __get_index_path(output_path))
        weights = weights_by_file(weights, index_path)
        for file_path, weights_file in weights.items():
            with open(os.path.join(output_path, file_path), "wb") as f:
                f.write(save(weights_file))
    else:
        if not output_path.endswith(".safetensors"):
            os.makedirs(output_path, exist_ok=True)
            save_path = os.path.join(output_path, "model.safetensors")
        else:
            save_path = output_path
        with open(save_path, 'wb') as f:
            f.write(save(weights))
