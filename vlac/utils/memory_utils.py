from typing import Dict, Callable, List

import torch
import torch.nn as nn
import math


def format_bytes(bytes_value):
    """
    Format bytes into a human-readable string with appropriate unit.

    Args:
        bytes_value: The value in bytes

    Returns:
        A string with the formatted value and unit
    """
    if bytes_value == 0:
        return "0 B"

    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(bytes_value, 1024)))
    p = math.pow(1024, i)
    s = round(bytes_value / p, 2)

    return f"{s} {size_name[i]}"


class GpuMemory:
    def __init__(self, device=None):
        self.device = torch.cuda.current_device() if device is None else device
        self.__processTotal()
        self.__processUsed()

    def __processTotal(self):
        self.total = torch.cuda.get_device_properties(self.device).total_memory
        self.total_human = format_bytes(self.total)

    def __processUsed(self):
        self.used = torch.cuda.memory_allocated(self.device)
        self.used_human = format_bytes(self.used)
        self.percentage = (self.used / self.total) * 100

    def refreshUsed(self):
        self.__processUsed()

    def __str__(self):
        return f"GPU Memory Usage ({self.device}): {self.used_human} / {self.total_human} ({self.percentage:.2f}%)"

    gpus: Dict[str, "GpuMemory"] = {}

    @classmethod
    def init_gpus(cls):
        gpus = {f"cuda:{i}": cls(f"cuda:{i}") for i in range(torch.cuda.device_count())}
        gpus["cuda"] = cls(torch.cuda.current_device())
        cls.gpus = gpus

    @staticmethod
    def of(device):
        mem = GpuMemory.gpus[str(device)]
        mem.refreshUsed()
        return mem


GpuMemory.init_gpus()


def print_gpu_memory_usage(device=None, label="Current"):
    """
    Print the current GPU memory usage in a human-readable format.
    
    Args:
        device: The device to check. If None, uses the current device.
        label: A label to identify this memory check in the output.
    """
    print(f'[{label}] {GpuMemory.of(device)}')


def print_gpus_memory_usage(label="Current"):
    for i in range(torch.cuda.device_count()):
        print_gpu_memory_usage(f"cuda:{i}", label)


def track_gpu_memory_usage_for_func(func: Callable, device=None, *args, **kwargs):
    """
    Track GPU memory usage before and after executing a function.
    
    Args:
        func: The function to execute
        device: The device to check. If None, use the current device.
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        The result of the function
    """
    mem = GpuMemory.of(device)
    before_used = mem.used
    print(f"[Before {func.__name__}] {mem}")

    # Execute function
    result = func(*args, **kwargs)

    # Get memory usage after
    mem.refreshUsed()
    diff = mem.used - before_used
    diff_human = format_bytes(diff)

    print(f"[After {func.__name__}] {mem}")
    print(f"[{func.__name__}] GPU Memory Increase: {diff_human}")

    return result


class MemoryOfModule:
    def __init__(self, module: nn.Module):
        self.module = module
        self.param_size = sum(p.numel() * p.element_size() for p in module.parameters(recurse=True))
        self.buffer_size = sum(b.numel() * b.element_size() for b in module.buffers(recurse=True))
        self.total_size = self.param_size + self.buffer_size
        first_param = next(module.parameters(), None)
        self.device = module.device if hasattr(module, "device") else first_param.device if first_param is not None else None

    def __str__(self):
        return f'{format_bytes(self.param_size)} param + {format_bytes(self.buffer_size)} buffer = {format_bytes(self.total_size)} ({self.device})'


def __track_memory_usage_for_model(model: nn.Module, max_depth: int, no_inside_modules: List[str], depth: int = 0) -> Dict | MemoryOfModule:
    if (max_depth != -1 and depth >= max_depth) or len(model._modules) == 0 or model._get_name() in no_inside_modules:
        return MemoryOfModule(model)
    track = {}
    for _, module in model._modules.items():
        if module is None:
            continue
        track[module._get_name()] = __track_memory_usage_for_model(module, max_depth, no_inside_modules, depth + 1)
    return track


def track_memory_usage_for_model(model: nn.Module, max_depth: int = -1, submodules: Dict[str, int] = None, no_inside_modules: List[str] = None) -> Dict | MemoryOfModule:
    if no_inside_modules is None: no_inside_modules = []
    if submodules is None: return __track_memory_usage_for_model(model, max_depth, no_inside_modules)
    track = {}
    for submodule, _max_depth in submodules.items():
        if _max_depth == -2:
            _max_depth = max_depth
        keys = submodule.split(".")
        current = track
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = __track_memory_usage_for_model(model.get_submodule(submodule), _max_depth, no_inside_modules)
    return track


def __print_memory_usage_from_track(track: Dict, prefix: str = ""):
    for key, value in track.items():
        if isinstance(value, dict):
            print(f'{prefix}{key}:')
            __print_memory_usage_from_track(value, f'{prefix}  ')
        else:
            print(f'{prefix}{key}: {value}')


def print_memory_usage_for_model(model: nn.Module, max_depth: int = -1, submodules: Dict[str, int] = None, no_inside_modules: List[str] = None, header_len: int = 64):
    base_header = f"Memory Usage of Model {model._get_name()}"
    header_pad = "=" * ((header_len - len(base_header)) // 2) if len(base_header) < header_len else ""
    print(f"{header_pad}{base_header}{header_pad}")
    __print_memory_usage_from_track(track_memory_usage_for_model(model, max_depth, submodules, no_inside_modules))
    print(f'\n{model._get_name()}: {MemoryOfModule(model)}')
    print("="*header_len)
