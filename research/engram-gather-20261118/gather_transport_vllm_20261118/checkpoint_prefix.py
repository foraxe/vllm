"""Read actual interleaved Engram rows from a local safetensors checkpoint."""
import json
import struct

import torch


def read_prefix(model, layer, rows):
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    table = torch.empty((rows, 264), dtype=torch.uint8)
    for kind, width, col, dtype in [('weight', 256, 0, 'F8_E4M3'), ('scale', 8, 256, 'F8_E8M0')]:
        name = f'layers.{layer}.engram.embed.{kind}'
        with (model / index[name]).open('rb', buffering=0) as f:
            size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(size)); spec = header[name]
            if spec['dtype'] != dtype or spec['shape'][1] != width or rows > spec['shape'][0]:
                raise ValueError('checkpoint table format mismatch')
            f.seek(8 + size + spec['data_offsets'][0]); raw = bytearray(f.read(rows * width))
        if len(raw) != rows * width:
            raise ValueError('checkpoint table is truncated')
        table[:, col:col + width].copy_(torch.frombuffer(raw, dtype=torch.uint8).view(rows, width))
    return table
