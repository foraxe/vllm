"""Matched offline engine performance for the reviewed DSV4.1 DCP branch."""
import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.renderers.params import ChatParams
from vllm.v1.worker.gpu_worker import Worker

ROOT = Path('/workspace/vllm_dsv41')

class PerfWorker(Worker):
    def load_model(self, **kwargs):
        self.record_steps = False
        self.steps = []
        self.seen = {}
        super().load_model(**kwargs)

    def execute_model(self, scheduler_output):
        if self.record_steps and self.rank == 0 and scheduler_output.total_num_scheduled_tokens:
            counts = scheduler_output.num_scheduled_tokens
            self.steps.append(sorted((self.seen.get(key, 0), count) for key, count in counts.items()))
            for key, count in counts.items():
                self.seen[key] = self.seen.get(key, 0) + count
        return super().execute_model(scheduler_output)

    def begin_run(self):
        self.steps = []
        self.seen = {}
        self.record_steps = True

    def end_run(self):
        self.record_steps = False
        return {'rank': self.rank, 'steps': self.steps,
                'runner': type(self.model_runner).__module__,
                'graph_mode': str(self.vllm_config.compilation_config.cudagraph_mode)}

    def begin_profile(self, prefix):
        if self.rank == 0:
            from profile_support import start_profile
            return start_profile(prefix + '.trace.json', prefix + '.summary.json', rank=self.rank)

    def end_profile(self):
        if self.rank == 0:
            from profile_support import stop_profile
            return stop_profile(rank=self.rank)


def make_prompts(llm, length, batch, retrieval=False):
    tokenizer = llm.get_tokenizer()
    notes = ''.join(
        f'Notebook entry {i}: measure GPU kernel duration, synchronize at workload boundaries, '
        f'preserve request lengths, and record the observation number {i % 97}.\n'
        for i in range(24000))
    filler = tokenizer.decode(tokenizer.encode(notes, add_special_tokens=False)[:length + 256])
    prompts = []
    for index in range(batch):
        message = [{'role': 'user', 'content': ('Remember verification code 654321.\n' if retrieval else '') + f'Notebook {index}.\n' + filler +
                    '\nWrite a detailed, extended explanation of how to make GPU inference '
                    'experiments reproducible. Use full paragraphs and examples.'}]
        if retrieval:
            message[0]['content'] = message[0]['content'].replace('Write a detailed, extended explanation of how to make GPU inference experiments reproducible. Use full paragraphs and examples.', 'What was the verification code at the beginning? Return only the code.')
        _, rendered = llm.llm_engine.renderer.render_chat(
            [message], ChatParams(chat_template_kwargs={'thinking': False, 'add_generation_prompt': True}))
        engine_input = rendered[0]
        ids = engine_input['prompt_token_ids']
        assert len(ids) >= length, (len(ids), length)
        # Remove only interior user text; retain the chat prefix and final question.
        ids = ids[:length - 128] + ids[-128:]
        assert len(ids) == length
        prompts.append({'prompt_token_ids': ids})
    return prompts


def run_batch(llm, prompts, output_tokens):
    params = SamplingParams(temperature=0, max_tokens=output_tokens,
                            min_tokens=output_tokens, ignore_eos=True)
    llm.sleep(level=0)
    llm.collective_rpc('begin_run')
    llm.enqueue(prompts, params, use_tqdm=False)
    start = time.perf_counter()
    llm.wake_up(tags=['scheduling'])
    outputs = llm.wait_for_completion(use_tqdm=False)
    wall = time.perf_counter() - start
    workers = llm.collective_rpc('end_run')
    rows = []
    for result in outputs:
        generated = result.outputs[0]
        assert len(generated.token_ids) == output_tokens
        assert result.metrics is not None
        metrics = dataclasses.asdict(result.metrics)
        assert metrics['num_preemptions'] == 0 and not metrics['is_corrupted']
        rows.append({'prompt_len': len(result.prompt_token_ids),
                     'tokens': generated.token_ids, 'text': generated.text,
                     'metrics': metrics,
                     'tpot_ms': 1000 * (metrics['last_token_ts'] - metrics['first_token_ts']) / (output_tokens - 1),
                     'ttft_ms': 1000 * metrics['first_token_latency']})
    steps = workers[0]['steps']
    signature = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    return {'wall_s': wall, 'output_tokens_per_s': len(rows) * output_tokens / wall,
            'mean_tpot_ms': statistics.fmean(x['tpot_ms'] for x in rows),
            'mean_ttft_ms': statistics.fmean(x['ttft_ms'] for x in rows),
            'scheduler_steps': steps, 'scheduler_signature': signature,
            'step_count': len(steps), 'outputs': rows, 'worker_modes': workers}


def main(args):
    cases = [(8192, 1), (32768, 1), (32768, 8)]
    if args.case:
        cases = [tuple(map(int, x.split(':'))) for x in args.case.split(',')]
    max_length = max(length for length, _ in cases)
    extra = {}
    if args.decode_graph:
        from vllm.config import CompilationMode, CUDAGraphMode
        maximum = max(batch for _, batch in cases)
        sizes = sorted(set([1, 2, 4, maximum] + list(range(8, maximum + 1, 8))))
        sizes = [size for size in sizes if size <= maximum]
        extra['compilation_config'] = {'mode': CompilationMode.NONE,
            'cudagraph_mode': CUDAGraphMode.FULL_DECODE_ONLY,
            'cudagraph_capture_sizes': sizes, 'max_cudagraph_capture_size': maximum}
    llm = LLM(model='/data/models/DeepSeek-V4.1-Flash', tokenizer_mode='deepseek_v41',
              tensor_parallel_size=4, decode_context_parallel_size=args.dcp,
              language_model_only=True, enforce_eager=not (args.graph or args.decode_graph),
              enable_prefix_caching=False, max_model_len=max_length + 1024,
              max_num_seqs=max(batch for _, batch in cases), max_num_batched_tokens=8192,
              kv_cache_memory_bytes=args.kv_gib * 2**30, gpu_memory_utilization=.55, seed=0,
              disable_log_stats=False,
              attention_config={'backend': 'FLASHMLA_SPARSE_DSV41', 'indexer_kv_dtype': 'fp8'},
              kernel_config={'enable_jit_warmup': False, 'enable_cutedsl_warmup': False,
                             'enable_flashinfer_autotune': False},
              worker_cls='benchmark.PerfWorker', **extra)
    report = {'dcp': args.dcp, 'graph_requested': args.graph or args.decode_graph, 'decode_graph': args.decode_graph, 'tp': 4,
              'output_tokens': args.output_tokens, 'kv_gib': args.kv_gib, 'repeats': args.repeats,
              'source_head': subprocess.check_output(['git', '-C', os.environ.get('DSV41_SOURCE_ROOT', str(ROOT/'vllm-dcp')), 'rev-parse', 'HEAD'], text=True).strip(),
              'case_results': []}
    path = Path(args.output)
    diff = subprocess.check_output(['git', '-C', os.environ.get('DSV41_SOURCE_ROOT', str(ROOT/'vllm-dcp')), 'diff', 'HEAD'], text=True)
    report['source_diff_sha256'] = hashlib.sha256(diff.encode()).hexdigest()
    path.with_suffix('.source.patch').write_text(diff)
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    save()
    for length, batch in cases:
        label = f'L{length}-B{batch}'
        prompts = make_prompts(llm, length, batch)
        prompt_hash = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        if args.check_retrieval:
            checks = llm.generate(make_prompts(llm, length, 1, retrieval=True), SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
            result = checks[0].outputs[0]
            report.setdefault('retrieval_checks', []).append({'context': length, 'text': result.text, 'tokens': result.token_ids})
            save()
            assert result.text.strip() == '654321', result.text
            print('RETRIEVAL_PASS ' + label, flush=True)
        print('WARMUP ' + label, flush=True)
        warmup = run_batch(llm, prompts, args.output_tokens)
        item = {'label': label, 'prompt_length': length, 'batch': batch,
                'prompt_sha256': prompt_hash, 'warmup': warmup, 'runs': []}
        report['case_results'].append(item)
        save()
        for repeat in range(args.repeats):
            result = run_batch(llm, prompts, args.output_tokens)
            item['runs'].append(result)
            save()
            print('RESULT ' + json.dumps({'case': label, 'repeat': repeat,
                **{k: result[k] for k in ['wall_s', 'mean_tpot_ms', 'mean_ttft_ms', 'output_tokens_per_s', 'step_count', 'scheduler_signature']}}), flush=True)
        if args.profile and (length, batch) == cases[-1]:
            prefix = str(path.with_suffix('')) + '-profile'
            llm.collective_rpc('begin_profile', args=(prefix,))
            item['profile_run'] = run_batch(llm, prompts, args.output_tokens)
            llm.collective_rpc('end_profile')
            save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcp', type=int, required=True)
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--decode-graph', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--check-retrieval', action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-tokens', type=int, default=64)
    parser.add_argument('--kv-gib', type=int, default=2)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
