import argparse
import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


class BenchWorker(Worker):
    def load_model(self, **kwargs):
        import torch
        import importlib.util
        import sys
        import vllm.model_executor.layers.sparse_attn_indexer as indexer
        if os.environ['CANDIDATE_VARIANT']=='baseline':
            from candidate_baseline import select_candidate_blocks
        else:
            spec=importlib.util.spec_from_file_location('candidate_optimized', '/workspace/vllm_dsv41/vllm-candidate-full/vllm/model_executor/kernels/attention/dsa/candidate_blocks.py')
            module=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=module
            spec.loader.exec_module(module)
            select_candidate_blocks=module.select_candidate_blocks
        self._impl=select_candidate_blocks
        self._calls={}
        self._capture_calls={}
        def count_selection(logits, starts, ends, k, block_size, out, row_repeat=1):
            counts=self._capture_calls if torch.cuda.is_current_stream_capturing() else self._calls
            key=str((tuple(logits.shape),k,block_size))
            counts[key]=counts.get(key,0)+1
            return self._impl(logits,starts,ends,k,block_size,out,row_repeat)
        indexer._select_candidate_blocks=count_selection
        super().load_model(**kwargs)
        assert self.vllm_config.use_v2_model_runner
        self._steps={'prefill':0,'decode':0,'idle':0,'tokens':[]}
        original_execute=self.model_runner.execute_model
        def count_step(scheduler_output,*args,**kw):
            lens=list(scheduler_output.num_scheduled_tokens.values())
            key='idle' if not sum(lens) else ('prefill' if max(lens)>1 else 'decode')
            self._steps[key]+=1
            self._steps['tokens'].append(sum(lens))
            return original_execute(scheduler_output,*args,**kw)
        self.model_runner.execute_model=count_step

    def probe(self,action):
        import copy
        import vllm.model_executor.layers.sparse_attn_indexer as indexer
        if action in ('reset','timing'):
            self._calls.clear()
            self._steps={'prefill':0,'decode':0,'idle':0,'tokens':[]}
        if action=='timing':indexer._select_candidate_blocks=self._impl
        return {'rank':self.rank,'calls':dict(self._calls),'capture_calls':dict(self._capture_calls),'steps':copy.deepcopy(self._steps)}


def main(args):
    import math
    from vllm import LLM,SamplingParams
    from vllm.tokenizers import get_tokenizer
    model='/data/models/DeepSeek-V4.1-Flash'
    llm=LLM(model=model,tokenizer_mode='deepseek_v41',tensor_parallel_size=4,
        language_model_only=True,max_model_len=40960,max_num_seqs=4,max_num_batched_tokens=8192,
        kv_cache_memory_bytes=2**30,enable_prefix_caching=False,seed=0,disable_log_stats=False,
        attention_config={'dsv4_fused_attention':False},
        compilation_config={'cudagraph_capture_sizes':[1,2,4,32,64]},
        kernel_config={'enable_jit_warmup':False,'enable_cutedsl_warmup':False,'enable_flashinfer_autotune':False},
        worker_cls='candidate_model_bench.BenchWorker')
    llm.collective_rpc('probe',args=('reset',))
    check=llm.chat([{'role':'user','content':'What is 17 times 19? Return only the integer.'}],
        SamplingParams(temperature=0,max_tokens=8),chat_template_kwargs={'thinking':False})[0]
    diagnostic=llm.collective_rpc('probe',args=('read',))
    assert check.outputs[0].text.strip()=='323',check.outputs[0].text
    print('MRV2_DIAGNOSTIC '+json.dumps(diagnostic),flush=True)
    tok=get_tokenizer(model,tokenizer_mode='deepseek_v41')
    short=list(check.prompt_token_ids)
    seed_ids=tok.encode('The sky is blue. ',add_special_tokens=False)
    prompts={17:short}
    assert len(short)==17
    for n in (8192,32768):prompts[n]=(seed_ids * math.ceil(n/len(seed_ids)))[:n]
    diagnostic_output=llm.generate({'prompt_token_ids':prompts[8192]},SamplingParams(temperature=0,max_tokens=8,ignore_eos=True),use_tqdm=False)[0].outputs[0].token_ids
    diagnostic=llm.collective_rpc('probe',args=('read',))
    assert all(x['calls'] for x in diagnostic),diagnostic
    print('CANDIDATE_DIAGNOSTIC '+json.dumps(diagnostic),flush=True)
    llm.collective_rpc('probe',args=('timing',))
    mixed=llm.generate([{'prompt_token_ids':short},{'prompt_token_ids':prompts[32768][:16384]}],
        SamplingParams(temperature=0,max_tokens=16,ignore_eos=True),use_tqdm=False)
    assert all(r.metrics is not None and not r.metrics.is_corrupted for r in mixed)
    mixed_outputs=[r.outputs[0].token_ids for r in mixed]
    results=[]
    path=Path(args.output)
    def save():path.write_text(json.dumps({'runner':'MRV2','variant':args.variant,'diagnostic':diagnostic,'diagnostic_output':diagnostic_output,'mixed_outputs':mixed_outputs,'results':results},indent=2))
    save()
    for n in args.contexts:
        for trial in range(args.trials+2):
            llm.collective_rpc('probe',args=('reset',))
            result=llm.generate({'prompt_token_ids':prompts[n]},SamplingParams(temperature=0,max_tokens=64,ignore_eos=True),use_tqdm=False)[0]
            stats=llm.collective_rpc('probe',args=('read',))
            assert all(x['steps']['prefill']==math.ceil(n/8192) and x['steps']['decode']==63 for x in stats),stats
            metrics=result.metrics
            assert metrics is not None and not metrics.is_corrupted and metrics.num_preemptions==0
            assert metrics.first_token_ts>metrics.scheduled_ts>0 and metrics.last_token_ts>metrics.first_token_ts
            row={'context':n,'trial':trial,'warmup':trial<2,'prefill_ms':(metrics.first_token_ts-metrics.scheduled_ts)*1000,
                 'ttft_ms':metrics.first_token_latency*1000,'tpot_ms':(metrics.last_token_ts-metrics.first_token_ts)/63*1000,
                 'token_ids':result.outputs[0].token_ids,'steps':stats[0]['steps']}
            assert len(row['token_ids'])==64
            results.append(row);save()
            print(json.dumps({k:v for k,v in row.items() if k not in ('token_ids','steps')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['baseline','candidate'],required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--contexts',nargs='+',type=int,default=[17,8192,32768]);parser.add_argument('--trials',type=int,default=6);parser.add_argument('--decode-min-tokens',type=int,default=1)
    args=parser.parse_args();os.environ['CANDIDATE_VARIANT']=args.variant;main(args)
