# Gathered Engram lookup: component experiment

A checkpoint-backed owner gathers requested table rows and sends them to a vLLM reader. The transport is **CPU-assisted**: GPU work signals a host worker, which arranges RDMA writes. This is a component experiment, not serving performance or a result for vLLM PR #64395. The reader is assembled by the harness; its owner lifecycle is stubbed for the timing run. No full-model inference is measured here.

## Matched results

Two H20 hosts, DOCA 3.5.0098, one owner and one reader, same checkpoint prefixes and input IDs. Each layer contains a 1,048,576-row prefix with 264 bytes per row. Fresh processes ran sequentially in native/gather/gather/native order, five warmups and 20 measured lookups per shape. Both arms kept native and gather resources allocated. Native uses grouped READ with depth 2; the gather arm selects gather at 32,768 rows, and the same native fallback below that threshold. Below 4,096 rows the fallback uses one QP; otherwise it uses 16.

| Rows | Native p50 (µs), two runs | Gather arm p50 (µs), two runs |
|---:|---:|---:|
| 64 | 159.2 / 159.1 | 165.6 / 164.0 |
| 1,024 | 945.2 / 945.9 | 952.9 / 957.6 |
| 4,096 | 773.5 / 774.9 | 782.8 / 783.2 |
| 32,768 | 4,686.9 / 4,707.5 | 1,225.9 / 1,213.5 |

The large-shape component latency is 3.8–3.9× lower in this experiment. The 32,768-row input includes 32,766 distinct valid IDs and two invalid IDs; partition ownership is checked separately. Event timing covers the lookup path including compaction, communication, wait and output handling, and can include host launch gaps. Reader process CPU and wall measurements are in the raw JSON; owner CPU was not isolated. The small-shape results exercise fallback, not gather, and do not establish the cause of the measured differences.

BF16 output from both layers was checked bit-for-bit including signed zeros, packed layer-14 output was checked byte-for-byte, and a two-layer outer CUDA graph was replayed with changing IDs. See the [four original sample files](results/). These checks are component checks; they do not establish model-output equivalence or end-to-end throughput.

## Sources and version boundaries

- The separately developed vLLM integration is commit `72279fb5de725e31864dc6c7b2e7a3e8f5b252a1`, based on PR head `31aaecaf18025bfe2808bf1883bc8a73834471cb`. The timing harness used the staged [provider source snapshot](provider-src/), which is preserved separately because subsequent fixes were applied to the integration branch.
- The preserved transport candidates are [original](gather_transport_vllm_20261118/gather_graph.cu) (SHA-256 `2f753a21d20cfccd1569d1822789ce2d47c76b5ea8301fc74719f92cf486bdc7`) and [v2](gather_transport_vllm_20261118/gather_graph.v2.cu) (SHA-256 `a48bb27abf18e661bddd9e3a935df5f2414a04dfd41dc7a2cd335fd8076d65b5`). Both existed before the timing runs; the retained timing artifacts do not independently identify which DSO was loaded, so an exact transport binary attribution is not asserted here. The later v2 error-path test is separate from the timing data.
- Native GET used the DOCA 3.5 research candidate. Its [source patches, build and dependencies](https://github.com/foraxe/nixl/tree/c666b177/research/gpunetio_doca35_get_closeout) are published separately. The runtime ABI and library paths must match. The two algorithms here are not a comparison with UCX or local table lookup.
- The execution lease and cleanup requirements are described in [CONTRACT.md](CONTRACT.md). Do not expose captured graphs to unmanaged replay. This experiment does not establish general crash or cancellation recovery.

## Reproduce the component comparison

Use two Linux hosts with compatible NVIDIA GPUs and RDMA, CUDA-enabled PyTorch, Triton, safetensors, CUDA toolkit and libibverbs. Build the compatible native GET runtime using the linked NIXL instructions. Both hosts need the same checkpoint and source package. Configure each host's `GPUNETIO_DEVICE_API_RDMA`, `GPUNETIO_DEVICE_API_GID_INDEX`, `GPUNETIO_DEVICE_API_OOB_INTERFACE`, matching NIXL plugin / library paths, and `GPUNETIO_NATIVE_QPS=1`; reserve separate control and native ports. Do not publish owner metadata: it contains memory keys and addresses. Use a trusted network.

Set the following separately on each host: `PKG` to this directory, `MODEL` to its local checkpoint, `NATIVE_SO` to the compatible native GET library, `RDMA_DEV` to its local RDMA device, and `GID` to its local GID index. Build the transport on both hosts:

```bash
cd "$PKG/gather_transport_vllm_20261118"
./build.sh
export GG_SO="$PWD/libgather_graph.so"
export SOURCE="$PKG/provider-src"
```

The build script builds the original candidate. To build v2 instead, copy `gather_graph.v2.cu` to `gather_graph.cu` in a disposable copy of this directory before building. Keep both hosts on the same variant and record source and DSO SHA-256 hashes. A successful rerun is needed to attribute results to either variant.

On the owner, set `OWNER_IP`, `GG_PORT`, `NATIVE_PORT`, and `RUN_DIR` to a fresh private directory reachable by the reader (or transfer the ready file securely); then run:

```bash
python "$PKG/gather_transport_vllm_20261118/lease_gpu_probe.py" \
  --role owner --library "$GG_SO" --native-library "$NATIVE_SO" \
  --source "$SOURCE" --native-port "$NATIVE_PORT" \
  --device "$RDMA_DEV" --gid "$GID" --host "$OWNER_IP" --port "$GG_PORT" \
  --model "$MODEL" --rows 1048576 --readers 4 --out "$RUN_DIR"
```

After `OWNER_READY`, run the following on the reader, using the owner's `OWNER_IP`, `GG_PORT`, and `RUN_DIR`, plus available `READER_NATIVE_PORT` and `GROUP_PORT_BASE`. The owner has a 300-second completion deadline; do not reuse a previous run directory.

```bash
for i in 0 1 2 3; do
  if [ "$i" = 0 ] || [ "$i" = 3 ]; then threshold=65536; else threshold=32768; fi
  python "$PKG/gather_transport_vllm_20261118/provider_checkpoint_probe.py" \
    --source "$SOURCE" --library "$GG_SO" --native-library "$NATIVE_SO" \
    --native-port "$READER_NATIVE_PORT" --native-qps 16 \
    --native-group-base "$GROUP_PORT_BASE" --model "$MODEL" \
    --owner-ready "$RUN_DIR/owner.ready" --host "$OWNER_IP" \
    --port "$((GG_PORT + i))" --device "$RDMA_DEV" --gid "$GID" \
    --reader-id "$i" --rows 1048576 --threshold "$threshold" \
    --pattern unique --samples 20 --out "$RUN_DIR/result-$i.json"
done
```

For hosts without shared storage, transfer each `reader-N.done` file back to the owner as well as securely transferring `owner.ready` to the reader. These scripts are the retained experimental harness; they require the matching runtime and a checkpoint with the expected table layout. Validate outputs, ports and source/binary identities for the selected environment.
