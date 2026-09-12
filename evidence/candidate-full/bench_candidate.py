import importlib.util,json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import select_candidate_blocks as candidate
from candidate_baseline import select_candidate_blocks as baseline
root=Path(__file__).parent
rows=[]
for n,width in [(1,10240),(64,10240),(1024,2048),(1024,8192),(1024,16384),(1024,32768),(8192,2048)]:
    logits=torch.randn(n,width,device='cuda')
    ends=torch.full((n,),width,device='cuda',dtype=torch.int32)
    a=torch.empty(n,2048,device='cuda',dtype=torch.int32);b=torch.empty_like(a)
    fa=lambda:baseline(logits,None,ends,2048,8,a)
    fb=lambda:candidate(logits,None,ends,2048,8,b)
    fa();fb()
    torch.testing.assert_close(a.sort().values,b.sort().values)
    times=[do_bench_cudagraph(f,rep=100)*1000 for f in (fa,fb,fb,fa)]
    result={'rows':n,'width':width,'a_us':(times[0]+times[3])/2,'b_us':(times[1]+times[2])/2,'abba_us':times}
    result['speedup']=result['a_us']/result['b_us'];rows.append(result)
    print(json.dumps(result),flush=True)
(root/'operator.json').write_text(json.dumps(rows,indent=2)+'\n')
