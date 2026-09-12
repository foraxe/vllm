"""Current-runtime DCP/automatic-cache checks and optional Engram ID traces."""
import argparse,json,re
from pathlib import Path
import numpy as np
import torch
from vllm import LLM,SamplingParams
from vllm.v1.worker.gpu_worker import Worker

ROOT=Path('/workspace/vllm_dsv41')

def extract(s):
    m=re.findall(r'####\s*(-?[\d,]+(?:\.\d+)?)',s)
    return m[-1].replace(',','') if m else None

class QualificationWorker(Worker):
    def load_model(self,**kwargs):
        from vllm.models.deepseek_v4_1.common.engram import ParallelEngramEmbedding
        from vllm.forward_context import get_forward_context
        original=ParallelEngramEmbedding.lookup
        self.trace_phase=None;self.traces={};self.trace_meta={}
        def lookup(layer,indices,out,background=False):
            if self.trace_phase is not None:
                meta=get_forward_context().attn_metadata
                nd=max((getattr(x,'num_decode_tokens',0) for x in meta.values()),default=0) if isinstance(meta,dict) else 0
                name=layer._trace_name
                ids=indices[:,layer.head_start:layer.head_start+layer.part_n_hash_cols].detach().cpu().numpy().copy()
                self.traces.setdefault(name,[]).append(ids)
                self.trace_meta.setdefault(name,[]).append({'phase':self.trace_phase,'decode_rows':nd,'rows':len(ids)})
            return original(layer,indices,out,background)
        ParallelEngramEmbedding.lookup=lookup
        super().load_model(**kwargs)
        self.embedding_meta={}
        for name,layer in self.model_runner.get_model().named_modules():
            if isinstance(layer,ParallelEngramEmbedding):
                layer._trace_name=name
                self.embedding_meta[name]={'head_start':layer.head_start,'heads':layer.part_n_hash_cols,'vocab_start':layer.vocab_start_idx,'vocab_end':layer.vocab_end_idx,'dim':layer.dim,'offloaded':layer.cpu_offload}
    def measure(self,reset=False):
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        if reset:torch.accelerator.reset_peak_memory_stats()
        s=torch.accelerator.memory_stats();free,total=torch.accelerator.get_memory_info()
        sizes={x.size for x in self.model_runner.kv_cache_config.kv_cache_tensors}
        assert len(sizes)==1
        kv=sizes.pop()
        result={'rank':self.rank,'allocated':s['allocated_bytes.all.current'],'peak':s['allocated_bytes.all.peak'],'free':free,'total':total,'init_free':self.init_snapshot.free_memory,'kv_bytes':kv,'profile_consumed':self.total_consumed,'profile_headroom':self.peak_activation_memory,'requested':self.requested_memory,'available_kv':self.available_kv_cache_memory_bytes}
        result['runtime_peak_estimate']=result['init_free']-free+result['peak']-result['allocated']
        result['profile_peak_plus_kv']=result['profile_consumed']+result['profile_headroom']+kv
        result['excess']=result['runtime_peak_estimate']-result['profile_peak_plus_kv']
        return result
    def trace(self,phase):self.trace_phase=phase;return self.rank
    def save_trace(self,label):
        self.trace_phase=None
        files=[]
        for i,(name,arrays) in enumerate(self.traces.items()):
            stem=ROOT/'artifacts/engram-traces'/f'{label}-rank{self.rank}-layer{i}'
            offsets=np.cumsum([0]+[len(a) for a in arrays])
            np.savez_compressed(stem.with_suffix('.npz'),ids=np.concatenate(arrays),offsets=offsets)
            stem.with_suffix('.json').write_text(json.dumps({'name':name,'embedding':self.embedding_meta[name],'calls':self.trace_meta[name]},indent=2)+'\n')
            files.append(str(stem))
        return {'rank':self.rank,'files':files}

def main(args):
    llm=LLM(model='/data/models/DeepSeek-V4.1-Flash',tokenizer_mode='deepseek_v41',
        tensor_parallel_size=4,decode_context_parallel_size=args.dcp,language_model_only=True,
        enforce_eager=True,enable_prefix_caching=False,max_model_len=32768,max_num_seqs=4,
        max_num_batched_tokens=args.batch_tokens,gpu_memory_utilization=.55,seed=0,
        attention_config={'backend':'FLASHMLA_SPARSE_DSV41','indexer_kv_dtype':'fp8'},
        kernel_config={'enable_jit_warmup':False,'enable_cutedsl_warmup':False,'enable_flashinfer_autotune':False},
        worker_cls='runtime_qualification.QualificationWorker')
    report={'dcp':args.dcp,'batch_tokens':args.batch_tokens,'start_memory':llm.collective_rpc('measure',args=(True,))}
    path=Path(args.output)
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    filler=''.join(f'Entry {i}: blue item {(i*37)%97}. ' for i in range(2400))
    prompts=[('What is 17 times 19? Return only the integer.','323'),
             ('Remember secret code 654321. '+filler+' What was the secret code? Return only the code.','654321'),
             ('Remember secret code 271828. '+filler+' X What was the secret code? Return only the code.','271828'),
             ('What is 13 times 7? Return only the integer.','91')]
    outputs=llm.chat([[{'role':'user','content':p}] for p,_ in prompts],SamplingParams(temperature=0,max_tokens=12),chat_template_kwargs={'thinking':False},use_tqdm=False)
    report['stress']=[{'prompt_len':len(r.prompt_token_ids),'text':r.outputs[0].text,'tokens':r.outputs[0].token_ids,'expected':e} for r,(_,e) in zip(outputs,prompts)]
    report['stress_memory']=llm.collective_rpc('measure')
    report['sizing_ok']=all(x['runtime_peak_estimate']<=x['requested']+64*2**20 for x in report['stress_memory'])
    save();print('MEMORY '+json.dumps({'sizing_ok':report['sizing_ok'],'excess_MiB':[x['excess']/2**20 for x in report['stress_memory']]}),flush=True)
    assert all(x['text'].strip()==x['expected'] for x in report['stress']),report['stress']
    if args.memory_only:
        return
    data=json.loads((ROOT/'artifacts/dcp/gsm8k-subset.json').read_text())
    demos='\n\n'.join('Question: '+x['question']+'\nAnswer: '+x['answer'] for x in data['splits']['train'])
    chats=[[{'role':'user','content':'Solve the final problem using the examples. End your answer with #### followed by the final number.\n\n'+demos+'\n\nQuestion: '+x['question']+'\nAnswer:'}] for x in data['splits']['test']]
    if args.trace:llm.collective_rpc('trace',args=('gsm_five_shot',))
    outputs=llm.chat(chats,SamplingParams(temperature=0,max_tokens=1024),chat_template_kwargs={'thinking':False},use_tqdm=False)
    rows=[]
    for i,(r,e) in enumerate(zip(outputs,data['splits']['test'])):
        o=r.outputs[0];answer=extract(o.text);expected=extract(e['answer'])
        rows.append({'id':i,'prompt_len':len(r.prompt_token_ids),'answer':answer,'expected':expected,'correct':answer==expected,'tokens':o.token_ids,'text':o.text,'finish_reason':o.finish_reason})
    report['gsm8k']=rows;report['correct']=sum(x['correct'] for x in rows);report['truncated']=sum(x['finish_reason']=='length' for x in rows)
    save();print('GSM8K '+json.dumps({'correct':report['correct'],'truncated':report['truncated']}),flush=True)
    if args.trace:
        llm.collective_rpc('trace',args=('gsm_zero_shot',))
        chats=[[{'role':'user','content':x['question']+' Give a concise solution.'}] for x in data['splits']['test'][:8]]
        llm.chat(chats,SamplingParams(temperature=0,max_tokens=128),chat_template_kwargs={'thinking':False},use_tqdm=False)
        report['traces']=llm.collective_rpc('save_trace',args=(f'dcp{args.dcp}-b{args.batch_tokens}',))
    report['final_memory']=llm.collective_rpc('measure');save()
    assert report['truncated']==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dcp',type=int,required=True);p.add_argument('--batch-tokens',type=int,default=8192);p.add_argument('--trace',action='store_true');p.add_argument('--memory-only',action='store_true');p.add_argument('--output',required=True);main(p.parse_args())
