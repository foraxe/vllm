import argparse,json,re
from pathlib import Path
from vllm import LLM,SamplingParams

def extract(s):
    matches=re.findall(r'####\s*(-?[\d,]+(?:\.\d+)?)',s)
    return matches[-1].replace(',','') if matches else None

def main(args):
    data=json.loads(Path(__file__).with_name('gsm8k-subset.json').read_text())
    llm=LLM(model='/data/models/DeepSeek-V4.1-Flash',tokenizer_mode='deepseek_v41',
            tensor_parallel_size=4,decode_context_parallel_size=args.dcp,
            language_model_only=True,enforce_eager=True,enable_prefix_caching=False,
            max_model_len=32768,max_num_seqs=4,max_num_batched_tokens=1024,
            kv_cache_memory_bytes=512*2**20,seed=0,attention_config={'backend':'FLASHMLA_SPARSE_DSV41'},
            kernel_config={'enable_jit_warmup':False,'enable_cutedsl_warmup':False,'enable_flashinfer_autotune':False})
    filler=''.join(f'Entry {i}: blue item {(i*37)%97}. ' for i in range(2400))
    message='Remember secret code 654321. '+filler+' What was the secret code? Return only the code.'
    long=llm.chat([{'role':'user','content':message}],SamplingParams(temperature=0,max_tokens=12),chat_template_kwargs={'thinking':False},use_tqdm=False)[0]
    assert len(long.prompt_token_ids)>16384
    long_result={'prompt_len':len(long.prompt_token_ids),'text':long.outputs[0].text,'tokens':long.outputs[0].token_ids}
    path=Path(args.output)
    report={'dcp':args.dcp,'dataset_revision':data['source_revision'],'long_retrieval':long_result,'results':[]}
    path.write_text(json.dumps(report,indent=2)+'\n')
    assert long.outputs[0].text.strip()=='654321',long_result
    demos='\n\n'.join('Question: '+x['question']+'\nAnswer: '+x['answer'] for x in data['splits']['train'])
    chats=[[{'role':'user','content':'Solve the final problem using the examples. End your answer with #### followed by the final number.\n\n'+demos+'\n\nQuestion: '+x['question']+'\nAnswer:'}] for x in data['splits']['test']]
    outputs=llm.chat(chats,SamplingParams(temperature=0,max_tokens=1024,logprobs=5),chat_template_kwargs={'thinking':False},use_tqdm=False)
    for i,(r,example) in enumerate(zip(outputs,data['splits']['test'])):
        out=r.outputs[0];expected=extract(example['answer']);actual=extract(out.text)
        row={'id':i,'prompt_len':len(r.prompt_token_ids),'expected':expected,'answer':actual,
             'correct':expected==actual,'text':out.text,'tokens':out.token_ids,'finish_reason':out.finish_reason,
             'logprobs':[{str(k):v.logprob for k,v in step.items()} for step in out.logprobs]}
        report['results'].append(row)
    report['correct']=sum(x['correct'] for x in report['results'])
    report['truncated']=sum(x['finish_reason']=='length' for x in report['results'])
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'dcp':args.dcp,'correct':report['correct'],'total':len(outputs),'truncated':report['truncated'],'long_retrieval':long_result}),flush=True)
    assert report['truncated']==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dcp',type=int,required=True);p.add_argument('--output',required=True);main(p.parse_args())
