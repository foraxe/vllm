# Shared Engram NVL72 two-host gate

Status: ready to run, not passed. Only one four-GPU GB200 host was available on
2026-10-27. The same-host probe below validates the script and local CUDA
mapping; it does not establish cross-host FABRIC import.

## Preparation

- Use two distinct hosts in one NVL72 fabric and NVIDIA IMEX channel. Record
  hostnames, GPU UUIDs, driver/CUDA/IMEX state, NVLink health and the owner and
  consumer source/runtime commits. Reserve the GPUs on both hosts. Stop if
  ownership or fabric health is ambiguous.
- Install the PR source and compatible native runtime on both hosts. Use the
  same DeepSeek-V4.1-Flash checkpoint, including identical `config.json` and
  `model.safetensors.index.json` bytes. Record their SHA256 values.
- Make a fresh private token file available to both hosts, and allow the
  consumer to reach an owner TCP address on the trusted cluster network. Do
  not put the token value in logs. The owner must have enough Grace memory for
  202,758,946,816 bytes of padded table backing.

## First gate: direct remote rows

On the owner host, with the PR worktree virtual environment active:

```bash
python -m vllm.models.deepseek_v41.nvidia.ops.shared_engram_owner \
  --model-dir /data/models/DeepSeek-V4.1-Flash \
  --listen-host OWNER_ADDRESS --listen-port 56000 \
  --token-file /shared/private/engram.token --device 0 \
  > owner.log 2>&1
```

Wait for `ENGRAM_OWNER_READY` and record its `generation=` value. Keep this
process running. On the **other** host, copy
`scripts/remote_consumer_probe.py.txt` to `remote_consumer_probe.py` and run:

```bash
python remote_consumer_probe.py \
  --model-dir /data/models/DeepSeek-V4.1-Flash \
  --endpoint OWNER_ADDRESS:56000 \
  --token-file /shared/private/engram.token \
  --generation GENERATION_ID --device 0 \
  --output remote_consumer.json > remote_consumer.log 2>&1
```

Accept only if the consumer hostname differs from the owner hostname, the
owner attach log records the consumer's non-loopback peer address, and both
layers report exact checkpoint equality for all seven selected boundary rows,
`graph_changed_ids=true`, no table-sized consumer RSS increase, and a normal
detach for every lease. Capture owner backing bytes, handles, placement and
generation. Shut down the owner only after the consumer exits and the detach
logs are complete; it must exit 0 with no uncertain lease. If import, lookup,
graph replay or fabric health fails, stop before full serving and retain both
logs and the CUDA/IMEX topology. In particular, a same-host pass is not a
substitute for this gate.

## Second gate: full P/D serving

With a fresh generation, place P TP2 on two GPUs of the owner host and D TP2
on two GPUs of the consumer host. Start each instance with the feature-guide
`--engram-config` pointing to the same owner endpoint and generation, lazy
safetensors, and the existing NIXL P/D KV connector. Record the two process
groups' hostnames and rank/GPU placement. Run the matched ordinary-table
control with the same model, inputs, KV settings, scheduler settings and
completion limit. Use the existing short P/D smoke first, then the bounded 8K
screen; admit 128K and 512K only after those pass and memory budgets permit.

Accept the MVP only if both independent instances finish the handoff with
matched prompt/output counts and completion text, the owner logs exactly one
physical table generation, neither consumer loads a full table replica, and
all leases detach cleanly. Report P and D latency separately with prompt
hashes, order, sample counts, and GPU health. The local 20% median regression
screen is a useful first bound; a cross-host performance claim needs repeated
matched measurements and cannot be inferred from the component probe.

If a consumer loses its socket while a mapping might still be used, retain
the owner: its uncertain lease intentionally prevents teardown. Establish
that the remote consumer process and GPU context are gone before forcing owner
recovery. Do not test active-graph owner death as part of this first gate.
