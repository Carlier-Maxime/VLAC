from argparse import ArgumentParser, Namespace
from copy import deepcopy


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
