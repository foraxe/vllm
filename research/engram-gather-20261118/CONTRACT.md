# Gather transport candidate: execution and retirement contract

This is an experimental transport contract. See [README](README.md) for measured versions, limitations, and reproduction.


## Execution lease

```
int gg_reader_begin_execution(void *reader);
int gg_reader_begin_execution_wait(void *reader, unsigned timeout_ms);
int gg_reader_end_execution(void *reader, uint64_t consumer_stream);
int gg_reader_abort_execution(void *reader);
```

Each framework worker must acquire the reader lease **before writing any shared IDs, hash/dedup workspace, static graph input, or launching a graph that uses the reader**. `begin` returns an error if a prior host lease is active, its last GPU consumer event is not yet complete, the reader is closing, or sticky transport error was observed. Use a distinct reader, workspace and captured graph for independent concurrent executions.

After scheduling the final payload consumer, the caller invokes `end` outside graph capture on a CUDA stream ordered after that consumer, including work on prefetch streams. `end` records an event; the next `begin` cannot succeed until it completes. The caller must continue holding all input/output buffers and the reader until that event is complete. The nonblocking `begin` fails if busy; `begin_execution_wait` waits on the host, up to the supplied positive timeout, for the prior host lease and GPU consumer event. It returns only once writing shared inputs is safe. The bounded wait's correctness and CPU/latency cost require hardware and serving validation.

`gg_fetch` requires an active lease at submission/capture time. **CUDA graph replay bypasses `gg_fetch` and all host code**: this C ABI cannot intercept a graph launched by an external caller. A framework must own and guard *every* replay path and never expose a graph for independent replay; otherwise the configuration is unsupported. A per-reader GPU lock inside `gg_fetch` is insufficient because input/workspace writes can occur earlier. In particular, this API alone does not prove concurrent FULL/PIECEWISE graph replay safe.

If submission or framework execution fails before a safe end, `abort_execution` marks the reader failed and keeps its ownership for quarantine. A cross-node bad-rkey graph replay with in-flight abort rejected subsequent begin and both close phases; the isolated processes retained their resources until exit. General crash or cancellation recovery remains unvalidated.

For graph capture, acquire before any shared-buffer writes, emit the graph's operations, finish capture, then end the host lease outside capture. For each subsequent replay, reacquire before updates and replay, then end after the true final consumer. Validate CUDA capture and cross-stream ordering on real hardware before use.

## Two-phase reader shutdown

```
int gg_reader_prepare_close(void *reader);
int gg_reader_close(void *reader);
```

The application first prevents new graph launches and releases any host execution lease. `prepare_close` blocks future API admission, synchronizes device work, checks sticky GPU/worker errors, completes the owner protocol and joins the worker. It keeps reader registrations, GPU buffers, control memory, and graph-owned resources allocated. On success, destroy graph references, then `close` deregisters and frees transport resources. On a preparation failure, retain the reader, all external tensors, graph objects, table backing and owner lease until safe retirement or process exit. A final cleanup error also poisons the reader and requires retaining remaining resources.

This contract does not provide crash recovery, peer-failure fencing, owner multi-reader qualification, or proof of vendor-specific GPUDirect write ordering. All must be validated separately.
