"""Matched SP Engram model validation; keeps all non-Engram configuration equal."""
import argparse,dataclasses,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/workspace/vllm_dsv41/artifacts/dcp/performance')
import benchmark as perf
from vllm import LLM,SamplingParams
ROOT=Path('/workspace/vllm_dsv41')

class TailWorker(perf.PerfWorker):
    def diagnostic(self):
        rows=[]
        for module in self.model_runner.get_model().modules():
            tail=getattr(module,'decoder_tail',None)
            if tail is not None:rows.append({'calls':tail.calls,'full_rows':tail.full_rows,'kept_rows':tail.kept_rows,'sp':tail.sequence_parallel})
        import deep_gemm
        from vllm.models.deepseek_v41.nvidia.ops.mega_mhc import is_mega_mhc_supported
        return {'rank':self.rank,'tail':rows,'deep_gemm':deep_gemm.__file__,'deep_gemm_version':deep_gemm.__version__,'mega_mhc':is_mega_mhc_supported(5120,4)}
    def end_run(self):
        value=super().end_run();value['tail']=self.diagnostic()['tail'];return value


def make_document_prompts(llm,lengths):
    from vllm.renderers.params import ChatParams
    repo=ROOT/'vllm-decoder-tail'
    docs=sorted((repo/'docs/design').glob('*.md'))
    documents=[p.read_text() for p in docs]
    tokenizer=llm.get_tokenizer()
    result=[]
    for i,length in enumerate(lengths):
        corpus='\n\n'.join(documents[(i*3+j)%len(documents)] for j in range(12))
        ids=tokenizer.encode(corpus,add_special_tokens=False)
        assert len(ids)>length+128
        content=tokenizer.decode(ids[:length+128])
        messages=[{'role':'user','content':f'Review packet {i}.\n'+content+'\nExplain the main engineering tradeoffs in these documents. Give a detailed answer with concrete examples.'}]
        _,rendered=llm.llm_engine.renderer.render_chat([messages],ChatParams(chat_template_kwargs={'thinking':False,'add_generation_prompt':True}))
        ids=rendered[0]['prompt_token_ids']
        ids=ids[:length-128]+ids[-128:]
        assert len(ids)==length
        result.append({'prompt_token_ids':ids})
    return result

def main(a):
    path=Path(a.output)
    report={'tail':a.tail,'source_head':subprocess.check_output(['git','-C',str(ROOT/'vllm-decoder-tail'),'rev-parse','HEAD'],text=True).strip(),'cases':[]}
    diff=subprocess.check_output(['git','-C',str(ROOT/'vllm-decoder-tail'),'diff','HEAD'],text=True)
    path.with_suffix('.source.patch').write_text(diff)
    report['source_diff_sha256']=hashlib.sha256(diff.encode()).hexdigest()
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    start=time.perf_counter()
    llm=LLM(model='/data/models/DeepSeek-V4.1-Flash',tokenizer_mode='deepseek_v41',
        tensor_parallel_size=4,enable_expert_parallel=True,language_model_only=True,
        enforce_eager=False,enable_prefix_caching=False,max_model_len=9216,
        max_num_seqs=64,max_num_batched_tokens=8192,kv_cache_memory_bytes=4*2**30,
        seed=0,disable_log_stats=False,
        attention_config={'indexer_kv_dtype':'fp8'},
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64]},
        kernel_config={'moe_backend':'deep_gemm_mega_moe','enable_jit_warmup':False,
          'enable_cutedsl_warmup':False,'enable_flashinfer_autotune':False},
        engram_config={'cpu_offload':True},additional_config={'dsv41_exact_decoder_tail':a.tail},
        worker_cls='tail_bench.TailWorker')
    report['engine_init_s']=time.perf_counter()-start
    report['diagnostic']=llm.collective_rpc('diagnostic');save()
    assert all(bool(d['tail'])==a.tail for d in report['diagnostic'])
    checks=[(f'What is {i} times {j}? Return only the integer.',str(i*j)) for i,j in [(17,19),(13,7),(23,11),(29,3),(12,12),(31,4),(9,27),(18,16)]]
    filler=''.join(f'Entry {i}: blue item {(i*37)%97}. ' for i in range(260))
    checks += [('Remember secret code '+c+'. '+filler+' What was the secret code? Return only the code.',c) for c in ['314159','271828']]
    outputs=llm.chat([[{'role':'user','content':q}] for q,_ in checks],SamplingParams(temperature=0,max_tokens=16,stop=['\n']),chat_template_kwargs={'thinking':False},use_tqdm=False)
    report['smoke']=[{'expected':v,'text':o.outputs[0].text,'tokens':o.outputs[0].token_ids} for o,(_,v) in zip(outputs,checks)];save()
    assert all(x['text'].strip()==x['expected'] for x in report['smoke']),report['smoke']
    for length,batch in [(8192,1),(8192,16),(8192,64),(-1,24),(512,16)]:
        lengths=([512,2048,8192]*8 if length==-1 else [length]*batch)
        prompts=make_document_prompts(llm,lengths)
        retrieval=llm.generate(perf.make_prompts(llm,max(lengths),1,retrieval=True),SamplingParams(temperature=0,max_tokens=16,stop=['\n']),use_tqdm=False)[0].outputs[0]
        report.setdefault('retrieval',[]).append({'length':length,'text':retrieval.text});save()
        assert retrieval.text.strip()=='654321',retrieval.text
        item={'length':length,'lengths':lengths,'batch':batch,'prompt_sha256':hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),'warmup':perf.run_batch(llm,prompts,64),'runs':[]}
        report['cases'].append(item);save()
        for repeat in range(3):
            result=perf.run_batch(llm,prompts,64);item['runs'].append(result);save()
            print('RESULT',length,repeat,result['output_tokens_per_s'],result['mean_ttft_ms'],result['mean_tpot_ms'],flush=True)

    # Exercise a prefill chunk followed by one token, then actual decode.
    check=llm.generate(perf.make_prompts(llm,8193,1,retrieval=True),SamplingParams(temperature=0,max_tokens=16,stop=['\n']),use_tqdm=False)[0]
    report['chunk_boundary_retrieval']=check.outputs[0].text;save()
    assert check.outputs[0].text.strip()=='654321'
    before=llm.collective_rpc('diagnostic')
    check=llm.generate(perf.make_prompts(llm,512,1),SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1),use_tqdm=False)[0]
    after=llm.collective_rpc('diagnostic')
    report['prompt_logprobs_fallback']={'before':before,'after':after,'returned':check.prompt_logprobs is not None};save()
    assert before==after and check.prompt_logprobs is not None
    dataset_path=ROOT/'artifacts/dcp/gsm8k-subset.json'
    data=json.loads(dataset_path.read_text())
    report['eval_dataset_sha256']=hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    report['eval_source_revision']=data['source_revision']
    demos='\n\n'.join('Question: '+x['question']+'\nAnswer: '+x['answer'] for x in data['splits']['train'])
    chats=[[{'role':'user','content':'Solve the final problem using the examples. End your answer with #### followed by the final number.\n\n'+demos+'\n\nQuestion: '+x['question']+'\nAnswer:'}] for x in data['splits']['test']]
    report['eval_prompt_sha256']=hashlib.sha256(json.dumps(chats).encode()).hexdigest()
    results=llm.chat(chats,SamplingParams(temperature=0,max_tokens=2048),chat_template_kwargs={'thinking':False},use_tqdm=False)
    import re
    def extract(text):
        matches=re.findall(r'####\s*(-?[\d,]+(?:\.\d+)?)',text)
        return matches[-1].replace(',','') if matches else None
    report['gsm8k']=[{'id':i,'expected':extract(expected['answer']),'answer':extract(result.outputs[0].text),
      'correct':extract(result.outputs[0].text)==extract(expected['answer']),
      'text':result.outputs[0].text,'tokens':result.outputs[0].token_ids,'finish_reason':result.outputs[0].finish_reason}
      for i,(result,expected) in enumerate(zip(results,data['splits']['test']))]
    report['gsm8k_correct']=sum(x['correct'] for x in report['gsm8k']);save()
    report['eval_truncated_ids']=[x['id'] for x in report['gsm8k'] if x['finish_reason']=='length'];save()
    print('GSM8K',report['gsm8k_correct'],len(report['gsm8k']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tail',action='store_true');p.add_argument('--output',required=True);main(p.parse_args())
