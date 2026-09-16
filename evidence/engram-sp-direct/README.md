# DeepSeek V4.1 Engram SP shared-memory evidence

Feature head: `acbdd4470823fd055bebe2b0340afdd2724812e5`.
Base: `ca542ee154e91c9a7757fb0b8f9d90ee578fb613`.

[Results and limitations](RESULTS.md) describe the complete matrix, accuracy
rechecks, tests, topology and startup/memory tradeoffs. Raw JSON includes generated
tokens, request metrics and scheduler signatures. The first delivery-direct run
hit its strict no-truncation assertion; it is deliberately preserved. Subsequent
matched 2048-token evaluations and the fixed two-repeat check are separate files.

`reproduce.tar.gz` contains the exact harness scripts and shared benchmark helper.
They preserve the validation host's paths and dependency overlays; adapt env.sh
and ROOT for another installation. Use the feature checkout as the vLLM source,
matching native binaries/dependencies, and obtain the evaluation subset described
in [DATASET.md](DATASET.md). No model weights or virtual environment are bundled.
Acquire GPU ownership before executing. The production tests are in the feature
branch and can be rerun directly with pytest as shown in RESULTS.md.

These are bounded offline-engine observations, not a general serving speedup or
broad accuracy proof. Human line review and relevant test rerun remain required
before upstream submission.

Raw generated text is preserved, including words cut off by the fixed token
budget. The evidence-only commit skips the spelling hook for those observations;
production code passed all applicable hooks.
