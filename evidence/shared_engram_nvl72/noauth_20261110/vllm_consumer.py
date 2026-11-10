"""Read the synthetic owner table through vLLM's token-free VMM importer."""

import argparse
import json
from pathlib import Path

import torch

from vllm.distributed.engram_share.vmm.owner import attach_table, detach_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.accelerator.set_device_index(1)
    connection, mapping, manifest = attach_table(
        args.endpoint,
        1,
        3,
        64,
        (3,),
        1,
        expected_generation=args.generation,
        checkpoint_dir=args.model_dir,
    )
    try:
        actual = mapping.cuda.cpu()
        expected = torch.tensor([56] * 128 + [64] * 64 + [127] * 6, dtype=torch.uint8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(json.dumps({"result": "pass", "mapped_bytes": actual.numel()}))
    finally:
        detach_table(connection, mapping, manifest["lease"])


if __name__ == "__main__":
    main()
