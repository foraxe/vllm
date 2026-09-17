import hashlib,json,math,time,re
from pathlib import Path
import ablate,tail_bench
from vllm import LLM,SamplingParams
from vllm.renderers.params import ChatParams,TokenizeParams

if __name__=='__main__':
    root=Path('/workspace/vllm_dsv41/artifacts/decoder-tail-diagnosis');path=root/'pr-qualification.json';report={}
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    llm=LLM(model='/data/models/DeepSeek-V4.1-Flash',tokenizer_mode='deepseek_v41',tensor_parallel_size=4,
        enable_expert_parallel=True,language_model_only=True,enforce_eager=False,enable_prefix_caching=False,
        max_model_len=10240,max_num_seqs=64,max_num_batched_tokens=32768,kv_cache_memory_bytes=4*2**30,
        seed=0,disable_log_stats=False,attention_config={'indexer_kv_dtype':'fp8'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64]},
        kernel_config={'moe_backend':'deep_gemm_mega_moe','enable_jit_warmup':False,'enable_cutedsl_warmup':False,'enable_flashinfer_autotune':False},
        engram_config={'cpu_offload':True},additional_config={'dsv41_exact_decoder_tail':True},worker_cls='ablate.AblateWorker')
    llm.collective_rpc('setup_ablation');llm.collective_rpc('set_interval',args=(19,256,False))
    def run(prompts,params):
        before=llm.collective_rpc('diagnostic');llm.sleep(level=0);llm.collective_rpc('begin_run')
        llm.enqueue(prompts,params,use_tqdm=False);llm.wake_up(tags=['scheduling']);out=llm.wait_for_completion(use_tqdm=False)
        workers=llm.collective_rpc('end_run');after=llm.collective_rpc('diagnostic')
        return out,{'before':before,'after':after,'steps':workers[0]['steps']}
    prompts=tail_bench.perf.make_prompts(llm,8192,4,retrieval=True)
    out,route=run(prompts,SamplingParams(temperature=0,max_tokens=16,stop=['\n']))
    report['active_retrieval']={'route':route,'text':[o.outputs[0].text for o in out]};save()
    assert all(x.strip()=='654321' for x in report['active_retrieval']['text'])
    assert route['after'][0]['tail'][0]['calls']>route['before'][0]['tail'][0]['calls']
    out,route=run(prompts,SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1))
    report['long_prompt_logprobs']={'route':route,'counts':[len(o.prompt_logprobs) for o in out]};save()
    assert max(sum(n for _,n in step) for step in route['steps'])>=16384
    assert route['before']==route['after']
    for o in out:
        assert len(o.prompt_logprobs)==8192
        assert all(values is not None and o.prompt_token_ids[i] in values and math.isfinite(values[o.prompt_token_ids[i]].logprob) for i,values in enumerate(o.prompt_logprobs) if i>0)
    data_path=Path('/workspace/vllm_dsv41/artifacts/dcp/gsm8k-subset.json');data=json.loads(data_path.read_text())
    report['dataset_sha256']=hashlib.sha256(data_path.read_bytes()).hexdigest()
    demos='\n\n'.join('Question: '+x['question']+'\nAnswer: '+x['answer'] for x in data['splits']['train'])
    appendix='Reference note: the material in this appendix is unrelated to the math problem.\n'*1000
    chats=[[{'role':'user','content':'Ignore this unrelated reference appendix when solving the math problem.\n'+appendix+'\nEnd of appendix.\nSolve the final problem using the examples. End your answer with #### followed by the final number.\n\n'+demos+'\n\nQuestion: '+x['question']+'\nAnswer:'}] for x in data['splits']['test'][:8]]
    _,rendered=llm.llm_engine.renderer.render_chat(chats,ChatParams(chat_template_kwargs={'thinking':False,'add_generation_prompt':True}),TokenizeParams(max_total_tokens=None))
    assert all(len(r['prompt_token_ids'])>8192 for r in rendered)
    prompts=[{'prompt_token_ids':r['prompt_token_ids'][:128]+r['prompt_token_ids'][-8064:]} for r in rendered]
    (root/'long-eval-prompts.json').write_text(json.dumps(prompts)+'\n')
    report['long_eval_prompt_sha256']=hashlib.sha256(json.dumps(prompts).encode()).hexdigest();report['long_eval']={};save()
    def extract(text):
        m=re.findall(r'####\s*(-?[\d,]+(?:\.\d+)?)',text)
        return m[-1].replace(',','') if m else None
    for mode in [0,19]:
        llm.collective_rpc('set_interval',args=(mode,256,False))
        out,route=run(prompts,SamplingParams(temperature=0,max_tokens=2048))
        rows=[{'id':i,'expected':extract(e['answer']),'answer':extract(o.outputs[0].text),'correct':extract(e['answer'])==extract(o.outputs[0].text),'text':o.outputs[0].text,'tokens':o.outputs[0].token_ids,'finish_reason':o.outputs[0].finish_reason} for i,(e,o) in enumerate(zip(data['splits']['test'][:8],out))]
        report['long_eval'][str(mode)]={'rows':rows,'correct':sum(r['correct'] for r in rows),'route':route};save()
        assert all(r['finish_reason']!='length' for r in rows)
        if mode:assert route['after'][0]['tail'][0]['calls']>route['before'][0]['tail'][0]['calls']
        print('LONG_GSM',mode,report['long_eval'][str(mode)]['correct'],8,flush=True)
    report['complete']=True;save();print('QUALIFICATION_COMPLETE',flush=True)
