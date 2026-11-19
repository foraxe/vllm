"""Load vLLM provider source in an isolated component-test namespace."""
import importlib
import sys
import types


def setup(source):
    root = source / 'vllm'
    for name, path in [('vllm', root), ('vllm.distributed', root / 'distributed'),
                       ('vllm.distributed.engram_share', root / 'distributed/engram_share'),
                       ('vllm.distributed.engram_share.rdma', root / 'distributed/engram_share/rdma')]:
        module = types.ModuleType(name); module.__path__ = [str(path)]; sys.modules[name] = module
    return importlib.import_module('vllm.distributed.engram_share.rdma.runtime')
