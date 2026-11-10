# Engram VMM trusted-TCP no-token gate (2026-11-10)

Source heads: vLLM #64082 `19b053f6a3cdb6d0b051527fa1674da1ddf85f01`,
SGLang #47478 `62283405ade5a1d7857781a6b8dea7628f23a361`. Both remove the
shared token-file field, owner CLI option, and attach-message token. The owner
still requires a generation and checkpoint identity match, and retains leases
until imported GPU mappings detach.

Hardware: one GB200 host, GPUs 0 and 1 for the synthetic gate. Both were
leased and idle before the run; all processes exited and the lease was
released. The generated checkpoint contains one 198-byte FP8/E8M0 table,
not the real DeepSeek-V4.1-Flash checkpoint.

Run from the SGLang PR worktree with its Python environment:

```bash
PYTHONPATH=python .venv/bin/python \
  /path/to/vllm-evidence/evidence/shared_engram_nvl72/noauth_20261110/run_sglang_external_smoke.py
```

The script starts the vLLM owner in a separate process, then attaches one
SGLang consumer and one vLLM consumer sequentially to the same generation.
SGLang gathered all three expected rows and replayed a CUDA Graph with changed
IDs; vLLM read the expected 198 bytes through its imported GPU mapping. Owner
logs show two attaches and two normal detaches, no uncertain lease, and exit 0.
The script uses local worktree paths that should be adjusted on another host.
Focused CPU tests passed: vLLM 60 selected Engram/config/loader, 38 direct
reload, then 11 owner tests after the final draining-error wording change;
SGLang seven external and two history tests. Applicable changed-file hooks
passed in both repositories. The source PRs include a launch recipe with no
token file.

The full-model P TP2 + D TP2 rerun was attempted but stopped before serving:
the owner could not allocate the 188.833 GiB table because Grace NUMA node 1
had only about 14 GiB physically free when allocation reached 46.0/94.4 GiB
for the first table. This is an admission/resource failure; it provides no
full-model result for these heads. Earlier full-model serving evidence applies
to the prior token-authenticated VMM revision. Cross-host NVL72 remains
untested because only one host is available.
