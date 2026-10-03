"""Stub of gsplat.distributed: single process only, as MineGS always runs it."""


def cli(fn, args, verbose=False):
    return fn(0, 0, 1, args)
