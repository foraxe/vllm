import argparse,json
from pathlib import Path
from vllm import LLM,SamplingParams
from vllm.v1.worker.gpu_worker import Worker

class DCPWorker(Worker):
    def check_runner(self):
        assert self.vllm_config.use_v2_model_runner
        from vllm.v1.kv_cache_interface import iter_layer_specs
        groups=[]
        for g in self.model_runner.kv_cache_config.kv_cache_groups:
            specs=[]
            for spec in iter_layer_specs(g.kv_cache_spec):
                specs.append({'spec':type(spec).__name__,
                              'placement':str(getattr(spec,'dcp_kv_cache_placement',None)),
                              'ratio':str(spec.tokens_per_state),'block_size':spec.block_size})
            groups.append({'layers':g.layer_names,'specs':specs})
        return {'rank':self.rank,'groups':groups}

def main(args):
    llm=LLM(model='/data/models/DeepSeek-V4.1-Flash',tokenizer_mode='deepseek_v41',
            tensor_parallel_size=4,decode_context_parallel_size=args.dcp,
            language_model_only=True,enforce_eager=True,enable_prefix_caching=False,
            max_model_len=8192,max_num_seqs=4,max_num_batched_tokens=1024,
            kv_cache_memory_bytes=512*2**20,seed=0,attention_config={'backend':'FLASHMLA_SPARSE_DSV41'},
            kernel_config={'enable_jit_warmup':False,'enable_cutedsl_warmup':False,'enable_flashinfer_autotune':False},
            worker_cls='model_check.DCPWorker')
    groups=llm.collective_rpc('check_runner')
    filler=''.join(f'Entry {i}: blue item {(i*37)%97}. ' for i in range(260))
    prompts=[('What is 17 times 19? Return only the integer.','323'),
             ('What is 13 times 7? Return only the integer.','91'),
             ('Remember secret code 314159. '+filler+' What was the secret code? Return only the code.','314159'),
             ('Remember secret code 271828. '+filler+' X. What was the secret code? Return only the code.','271828')]
    output=llm.chat([[{'role':'user','content':p}] for p,_ in prompts],
                    SamplingParams(temperature=0,max_tokens=12),chat_template_kwargs={'thinking':False},use_tqdm=False)
    rows=[]
    for r,(_,expected) in zip(output,prompts):
        row={'prompt_len':len(r.prompt_token_ids),'text':r.outputs[0].text,
             'tokens':r.outputs[0].token_ids,'expected':expected}
        rows.append(row)
    result={'dcp':args.dcp,'groups':groups,'outputs':rows}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)
    assert all(row['text'].strip()==row['expected'] for row in rows),rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dcp',type=int,required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
