import argparse,json,hashlib,time,subprocess,os
from pathlib import Path
import tail_bench
from vllm import LLM, SamplingParams
SOURCE=os.environ.get('DSV41_SOURCE_ROOT','/workspace/vllm_dsv41/vllm-decoder-tail')

class AblateWorker(tail_bench.TailWorker):
    def setup_ablation(self):
        self.tail_model=next(m for m in self.model_runner.get_model().modules() if getattr(m,'decoder_tail',None) is not None)
        self.saved_tail=self.tail_model.decoder_tail
        from vllm.distributed import get_tp_group
        self.communicator=get_tp_group().device_communicator
        self.original_gather=self.communicator.custom_all_gather
        self.original_scatter=self.communicator.custom_reduce_scatter
    def set_interval(self,interval,alignment=1,force_nccl=False):
        self.tail_model.decoder_tail=self.saved_tail if interval else None
        self.saved_tail.compaction_interval=interval or 1
        self.saved_tail.row_alignment=alignment
        self.communicator.custom_all_gather=(lambda x:None) if force_nccl else self.original_gather
        self.communicator.custom_reduce_scatter=(lambda x:None) if force_nccl else self.original_scatter
        return self.rank

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--matrix',action='store_true');p.add_argument('--batch-tokens',type=int,default=8192);p.add_argument('--wide-only',action='store_true');p.add_argument('--followup',action='store_true');p.add_argument('--confirm',action='store_true');a=p.parse_args()
    path=Path(a.output);report={'max_num_batched_tokens':a.batch_tokens,'cases':[],'source_head':subprocess.check_output(['git','-C',SOURCE,'rev-parse','HEAD'],text=True).strip()}
    diff=subprocess.check_output(['git','-C',SOURCE,'diff'],text=True)
    path.with_suffix('.patch').write_text(diff);report['patch_sha256']=hashlib.sha256(diff.encode()).hexdigest()
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    llm=LLM(model='/data/models/DeepSeek-V4.1-Flash',tokenizer_mode='deepseek_v41',tensor_parallel_size=4,
        enable_expert_parallel=True,language_model_only=True,enforce_eager=False,enable_prefix_caching=False,
        max_model_len=9216,max_num_seqs=64,max_num_batched_tokens=a.batch_tokens,kv_cache_memory_bytes=4*2**30,
        seed=0,disable_log_stats=False,attention_config={'indexer_kv_dtype':'fp8'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64]},
        kernel_config={'moe_backend':'deep_gemm_mega_moe','enable_jit_warmup':False,'enable_cutedsl_warmup':False,'enable_flashinfer_autotune':False},
        engram_config={'cpu_offload':True},additional_config={'dsv41_exact_decoder_tail':True},worker_cls='ablate.AblateWorker')
    llm.collective_rpc('setup_ablation')
    report['diagnostic']=llm.collective_rpc('diagnostic');save()
    cases=[(8192,4)] if not a.matrix else [(8192,1),(8192,16),(8192,64),(-1,24),(512,16)]
    if a.wide_only:cases=[(8192,64)]
    if a.followup:cases=[(8192,1),(8192,16),(-1,24),(512,16)]
    if a.confirm:cases=[(8192,1),(8192,16),(8192,64),(-1,24),(512,16)]
    for length,batch in cases:
        prompts=tail_bench.make_document_prompts(llm,[512,2048,8192]*8 if length==-1 else [length]*batch)
        modes=[(0,1,False),(19,256,False),(19,256,False),(0,1,False)] if not a.matrix else [(0,1,False),(19,256,False)]
        if a.confirm and batch==64:modes=[(0,1,False),(19,256,False),(19,256,False),(0,1,False)]
        for interval,alignment,force in modes:
            llm.collective_rpc('set_interval',args=(interval,alignment,force))
            item={'length':length,'batch':batch,'interval':interval,'alignment':alignment,'force_nccl':force,'prompt_sha256':hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),'warmups':[],'runs':[]}
            report['cases'].append(item);save()
            outputs=64 if a.matrix else 4
            for _ in range(2):item['warmups'].append(tail_bench.perf.run_batch(llm,prompts,outputs));save()
            for _ in range(3):
                r=tail_bench.perf.run_batch(llm,prompts,outputs);item['runs'].append(r);save()
                print('RESULT',length,batch,interval,alignment,force,r['wall_s'],r['output_tokens_per_s'],flush=True)
    if a.followup:
        print('COMPLETE',flush=True)
        raise SystemExit(0)
    llm.collective_rpc('set_interval',args=(19,256,False))
    checks=[(f'What is {i} times {j}? Return only the integer.',str(i*j)) for i,j in [(17,19),(13,7),(23,11),(29,3),(12,12),(31,4),(9,27),(18,16)]]
    outputs=llm.chat([[{'role':'user','content':q}] for q,_ in checks],SamplingParams(temperature=0,max_tokens=16,stop=['\n']),chat_template_kwargs={'thinking':False},use_tqdm=False)
    report['smoke']=[{'expected':v,'text':o.outputs[0].text} for o,(_,v) in zip(outputs,checks)];save()
    assert all(x['text'].strip()==x['expected'] for x in report['smoke'])
    for n in [8192,8193]:
        out=llm.generate(tail_bench.perf.make_prompts(llm,n,1,retrieval=True),SamplingParams(temperature=0,max_tokens=16,stop=['\n']),use_tqdm=False)[0].outputs[0]
        report.setdefault('retrieval',[]).append({'length':n,'text':out.text});save()
        assert out.text.strip()=='654321'
    if a.wide_only or a.confirm:
        results=llm.generate(tail_bench.perf.make_prompts(llm,8192,4,retrieval=True),SamplingParams(temperature=0,max_tokens=16,stop=['\n']),use_tqdm=False)
        report['batched_retrieval']=[o.outputs[0].text for o in results];save()
        assert all(x.strip()=='654321' for x in report['batched_retrieval'])
    before=llm.collective_rpc('diagnostic')
    check=llm.generate(tail_bench.perf.make_prompts(llm,512,1),SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1),use_tqdm=False)[0]
    after=llm.collective_rpc('diagnostic')
    report['prompt_logprobs_fallback']={'before':before,'after':after,'returned':check.prompt_logprobs is not None};save()
    assert before==after and check.prompt_logprobs is not None
    dataset_path=Path('/workspace/vllm_dsv41/artifacts/dcp/gsm8k-subset.json')
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

    print('COMPLETE',flush=True)
