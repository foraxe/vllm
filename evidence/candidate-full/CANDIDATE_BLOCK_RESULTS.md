# V4.1 candidate selection when all blocks fit

## Result

Remove sorting and intermediate scores when topk_blocks >= ceil(logits_width / block_size). The existing reduction directly publishes candidate IDs, preserving the selected set, NaN/minus-infinity handling, newest-block inclusion, and -1 padding. Order is unspecified only on this path. The path that selects a strict subset retains its existing top-k order.

Nine focused GPU tests passed, including packed offsets/tails, eligibility boundaries, decode row repetition, output-stride guards, and changed-input graph replay. No model mathematics or logits values change.

## Operator evidence

GB200, CUDA-graph A/B/B/A timing, triton.testing.do_bench_cudagraph(rep=100), float32 logits, candidate budget 2048, block size 8. Both arms use exactly the production helper, with the baseline saved from main. Candidate membership matched in every shape.

| Rows | Logits width | Baseline us | Candidate us | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10240 | 25.507 | 1.481 | 17.22x |
| 64 | 10240 | 29.576 | 1.977 | 14.96x |
| 1024 | 2048 | 39.670 | 10.156 | 3.91x |
| 1024 | 8192 | 99.169 | 10.289 | 9.64x |
| 1024 | 16384 | 139.842 | 10.806 | 12.94x |
| 1024 | 32768 | 172.629 | 172.684 | 1.00x |
| 8192 | 2048 | 251.653 | 75.021 | 3.35x |

The 32768-wide case does not qualify for the fast path and is the unchanged-path control. The 10240-wide operator cases are not the decode shape in the model experiment below.

## Model qualification

MRV2, TP4 GB200, DeepSeek-V4.1-Flash at dba1be0a40aa45a94ad051997016db3960a90277. Non-fused attention, max_model_len=40960, max_num_batched_tokens=8192, max_num_seqs=4, 1 GiB KV/worker, no prefix cache, graph sizes [1,2,4,32,64], 64 output tokens, greedy, ignore_eos. A/B/B/A with two warmups and four measured requests per context/run.

All 72 outputs match token-for-token across arms. Mixed short/16K requests and 8K diagnostics also match. Every worker executes 1/1/4 prefill steps and 63 nonempty decode steps for 17/8192/32768 inputs. No preemptions or corrupted metrics. Selector shape instrumentation is removed before timing; scheduler instrumentation is identical in both arms.

Actual diagnostic shapes include [8192,8192] prefill (eligible) and [1,40960] full-graph decode (not eligible). Capture also exercises decode batches 2 and 4. All four workers show the same shapes.

| Context | Baseline TTFT ms | Candidate TTFT ms | Change | Baseline TPOT ms | Candidate TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 21.947 | 21.808 | -0.63% | 5.9404 | 5.9311 |
| 8192 | 168.429 | 164.851 | -2.12% | 6.4476 | 6.4355 |
| 32768 | 671.072 | 659.048 | -1.79% | 6.5142 | 6.4962 |

Serving TTFT is order-sensitive at every context; none improves in both order comparisons. No end-to-end speedup is claimed. TPOT pooled changes are below 0.3%; this is a bounded no-regression/correctness check. No broad accuracy, ROCm, DCP, DBO or speculative-decoding model qualification is asserted.

## Provenance and reproduction

Feature base: current main 46d2b23. Only candidate_blocks.py and its existing test suite are changed. The baseline helper is identical on this main and the older validated model integration.

Model runtime: ad72e296d3d0725395641b78273bc3820c38abc6, with fused attention disabled. A worker swaps only the source-selector helper between the saved baseline and the new implementation. This keeps the established Torch2.13/native/dependency stack coherent; today's main has unrelated native changes and was not mixed with old core binaries for full-model runs. Model evidence is on that integration, not a whole-model build of current main.

Acquire the shared GPU lease before reproducing GPU work. From /workspace/vllm_dsv41:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DEEP_GEMM_WARMUP=skip \
PYTHONPATH=$PWD/artifacts/candidate-full:$PWD/runtime-deps/nvidia_cutlass_dsl/dsl_packages:$PWD/runtime-deps:$PWD/vllm-fused-out \
vllm-fused-out/.venv/bin/python artifacts/candidate-full/candidate_model_bench.py \
  --variant baseline --contexts 17 8192 32768 --trials 4 --output artifacts/candidate-full/model-a1.json
```

Run run_remaining.py for B/B/A; summarize.py checks output equality and step counts. The model driver loads the optimized helper from vllm-candidate-full; adjust that path if moving the evidence. Operator command is bench_candidate.py with vllm-candidate-full on PYTHONPATH instead. Raw JSON, drivers, saved baseline, test logs and a SHA256 manifest accompany this report.

DCP remains a separate larger task: compressed indexer KV and candidate filtering are both explicitly unsupported upstream. Sparse candidate-consuming MQA integration belongs to #56254 and is not duplicated here.

Feature candidate: e43951fda1fafe7e29eb3788092834a6e0124180, based on 46d2b23ac5047a813ebb082122166e4ae09b5f39. Scoped pre-commit and mypy checks passed; commit hooks passed. GPUs were released after the last model run.
