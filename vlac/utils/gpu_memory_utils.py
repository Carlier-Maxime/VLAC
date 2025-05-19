from typing import Dict

import torch
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
    def init_gpus(cls) -> Dict[str, "GpuMemory"]:
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


def track_gpu_memory_usage(func, device=None, *args, **kwargs):
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
