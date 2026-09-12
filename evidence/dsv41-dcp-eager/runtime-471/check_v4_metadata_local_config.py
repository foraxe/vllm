"""Run the existing V4 metadata assertion with an accessible model config.

The default helper fetches gated Llama3 metadata. This changes only the helper's
model_name; the cache spec, builder, input tensors and assertions are unchanged.
No model weights are loaded.
"""
from tests.v1.attention import test_indexer_deepseek_v4_slot_mapping as test
from tests.v1.attention.utils import create_vllm_config

test.create_vllm_config = lambda **kwargs: create_vllm_config(
    model_name="facebook/opt-125m", **kwargs
)
test.test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_num_states()
print("PASS: existing V4 compressed-slot metadata assertion, OPT config fixture")
