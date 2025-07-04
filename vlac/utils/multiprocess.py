import os
from argparse import ArgumentParser, Namespace
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


def add_multiprocess_args(parser: ArgumentParser, max_ntasks: int = None) -> ArgumentParser:
    parser.add_argument("--procid", type=int, default=None)
    parser.add_argument("--ntasks", type=int, default=max_ntasks)
    return parser


def check_multiprocess_args(parser: ArgumentParser, args: Namespace, max_ntasks: int = None) -> Namespace:
    if args.ntasks is None:
        args.ntasks = 1 if max_ntasks is None else max_ntasks
    if args.ntasks < 1 or (max_ntasks is not None and args.ntasks > max_ntasks):
        parser.error(f"the ntasks must be between 1 and {max_ntasks}")
    if args.procid is not None and (args.procid < 0 or args.procid >= args.ntasks):
        parser.error("the procid must be between 0 and ntasks-1")
    args.master = args.procid is None or args.procid == 0
    print(f"task {args.procid} of {args.ntasks}")
    return args


def remove_multiprocess_args(args: Namespace) -> Namespace:
    new_args = deepcopy(args)
    try:
        delattr(new_args, 'procid')
        delattr(new_args, 'ntasks')
        delattr(new_args, 'master')
    except AttributeError:
        pass
    return new_args


def parallel_threads_apply(data, func, num_thread: int = os.cpu_count(), desc: str = '', disable_bar: bool = False):
    results = []
    with ThreadPoolExecutor(num_thread) as executor:
        futures = [executor.submit(func, chunk) for chunk in data]
        for future in tqdm(as_completed(futures), total=len(data), desc=desc, unit='sample', leave=False, disable=disable_bar):
            results.append(future.result())
    return results


def parallel_threads_apply_(args):
    return parallel_threads_apply(*args)


def parallel_apply(data, func, num_proc: int = os.cpu_count(), num_thread_per_proc: int = 1, desc: str = '', concat_results: bool = True) -> list | np.ndarray:
    chunks = np.array_split(data, min(len(data) // (8*num_thread_per_proc), 100*num_proc))
    args = [(chunk, func, num_thread_per_proc, desc, num_proc > 1) for chunk in chunks]
    with Pool(num_proc) as pool:
        results = list(tqdm(
            pool.imap(parallel_threads_apply_, args),
            total=len(chunks),
            desc=desc,
            unit='chunk',
            disable=num_proc == 1
        ))
    return np.concatenate(results) if concat_results else results
