# Experimental gathered-row Engram provider

This opt-in research path adds owner-side NIC gather to the existing remote
Engram provider. It uses a separate, privately built C ABI library and a CPU
worker per reader. The default local and native GET paths do not load it.

Start the existing checkpoint owner with `--gather-library`, `--gather-device`,
`--gather-gid-index`, `--gather-base-port`, and `--gather-readers`. Each reader
gets one distinct, single-use endpoint. Reserve enough endpoints for all model
workers; released endpoints cannot be reused by this prototype. Owner and
reader processes must use a trusted network.

Set `rdma_provider` to
`vllm.distributed.engram_share.rdma.gather_provider` in the Engram configuration.
Configure the existing `ENGRAM_SHARED_*` variables as in the README, plus
`ENGRAM_GATHER_LIBRARY`, `ENGRAM_GATHER_DEVICE`, and
`ENGRAM_GATHER_GID_INDEX`. `ENGRAM_GATHER_THRESHOLD` defaults to 32768 logical
queries. The default fallback uses the existing native reader; to enable the
experimental grouped native fallback, set `ENGRAM_GATHER_NATIVE_QPS=16` and
`ENGRAM_GATHER_NATIVE_PORT_BASE` to a separate range of 32 ports per local GPU.

Framework-owned model execution, capture, and replay must hold an execution
lease before mutating shared inputs and release it after the last consumer.
External graph launches that bypass this guard are unsupported. Normal shutdown
first quiesces the transport while graphs and registered buffers remain alive,
then releases graphs, registrations, and the owner lease. A transport error
quarantines those resources until safe retirement or process exit.

This is a research prototype. Checkpoint-wide multi-reader memory registration,
full serving correctness, cancellation, crash recovery, CPU cost, and performance
must be validated for the target environment.
