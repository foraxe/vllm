# Evaluation data

The harness uses the cached first 5 GSM8K training
examples and first 32 test questions, from openai/gsm8k
configuration main, revision `3101c7d5072418e28b9008a6636bde82a006892c`. The dataset itself
is not bundled. Store the same subset as artifacts/dcp/gsm8k-subset.json with
keys source_revision and splits (train/test lists of question/answer records).
Prompt and dataset hashes are recorded in each evaluation result.
