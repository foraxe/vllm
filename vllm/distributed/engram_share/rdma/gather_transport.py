# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bindings for the optional, separately built gathered-row transport."""

import ctypes as c
from pathlib import Path


class UnsafeRetirementError(RuntimeError):
    """A setup failure left registrations that must remain alive."""


def load_library(path):
    lib = c.CDLL(str(Path(path).expanduser().resolve(strict=True)))
    ptr, u32, u64 = c.c_void_p, c.c_uint, c.c_uint64
    lib.gg_last_error.restype = c.c_char_p
    signatures = {
        "gg_reader_create": ([c.c_int, c.c_char_p, c.c_int, u32, u64, u64, u32], ptr),
        "gg_reader_connect": ([ptr, c.c_char_p, u32], c.c_int),
        "gg_reader_begin_execution_wait": ([ptr, u32], c.c_int),
        "gg_reader_end_execution": ([ptr, u64], c.c_int),
        "gg_reader_abort_execution": ([ptr], c.c_int),
        "gg_fetch": (
            [ptr, u64, u32, u64, u32, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64],
            c.c_int,
        ),
        "gg_record_consumed": ([ptr, u64], c.c_int),
        "gg_reader_prepare_close": ([ptr], c.c_int),
        "gg_reader_close": ([ptr], c.c_int),
    }
    for name, (args, result) in signatures.items():
        func = getattr(lib, name)
        func.argtypes, func.restype = args, result
    return lib


def check(lib, result):
    if result:
        error = lib.gg_last_error()
        raise RuntimeError(error.decode(errors="replace") if error else "gather failed")
