import torch
import math


def get_gpu_memory_usage(device=None):
    """
    Get the current GPU memory usage in a human-readable format.
    
    Args:
        device: The device to check. If None, uses the current device.
        
    Returns:
        A tuple containing (used_memory_bytes, total_memory_bytes, used_memory_human, total_memory_human)
    """
    if device is None:
        device = torch.cuda.current_device()

    # Get memory usage in bytes
    used_memory = torch.cuda.memory_allocated(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory

    # Convert to human-readable format
    used_memory_human = format_bytes(used_memory)
    total_memory_human = format_bytes(total_memory)

    return used_memory, total_memory, used_memory_human, total_memory_human


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


def print_gpu_memory_usage(device=None, label="Current"):
    """
    Print the current GPU memory usage in a human-readable format.
    
    Args:
        device: The device to check. If None, uses the current device.
        label: A label to identify this memory check in the output.
    """
    if device is None:
        device = torch.cuda.current_device()

    used, total, used_human, total_human = get_gpu_memory_usage(device)
    percentage = (used / total) * 100

    print(f"[{label}] GPU Memory Usage: {used_human} / {total_human} ({percentage:.2f}%)")


def track_gpu_memory_usage(func, *args, **kwargs):
    """
    Track GPU memory usage before and after executing a function.
    
    Args:
        func: The function to execute
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        The result of the function
    """
    device = torch.cuda.current_device()

    # Get memory usage before
    before_used, _, before_used_human, _ = get_gpu_memory_usage(device)
    print(f"[Before {func.__name__}] GPU Memory Usage: {before_used_human}")

    # Execute function
    result = func(*args, **kwargs)

    # Get memory usage after
    after_used, total, after_used_human, total_human = get_gpu_memory_usage(device)
    diff = after_used - before_used
    diff_human = format_bytes(diff)
    percentage = (after_used / total) * 100

    print(f"[After {func.__name__}] GPU Memory Usage: {after_used_human} / {total_human} ({percentage:.2f}%)")
    print(f"[{func.__name__}] GPU Memory Increase: {diff_human}")

    return result
