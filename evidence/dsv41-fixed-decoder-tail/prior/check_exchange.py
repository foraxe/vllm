import json,socket,os
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from vllm.config import VllmConfig,set_current_vllm_config
from vllm.config.parallel import ParallelConfig
from vllm.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_model_parallel,destroy_distributed_environment
from vllm.models.common.ops.sequence_parallel import sp_shard
from vllm.models.deepseek_v41.decoder_tail import decoder_tail_rows,redistribute_sp_states

def worker(rank,world,port):
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(world),MASTER_ADDR='localhost',MASTER_PORT=str(port))
    cfg=VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    init_distributed_environment()
    with set_current_vllm_config(cfg):initialize_model_parallel(tensor_model_parallel_size=world)
    results=[]
    for lengths in [[4096],[1,4096,17],[8192,3,4097]]:
        starts=np.concatenate(([0],np.cumsum(lengths)));total=int(starts[-1]);mapping=np.arange(total)
        torch.manual_seed(21)
        full=[torch.randn(total,8,dtype=torch.bfloat16,device='cuda'),None,
              torch.arange(total,dtype=torch.int64,device='cuda'),
              torch.randn(total,4,dtype=torch.float32,device='cuda'),
              torch.randn(total,4,dtype=torch.float32,device='cuda'),
              torch.randn(total,4,4,dtype=torch.float32,device='cuda'),
              torch.randn(total,4,8,dtype=torch.bfloat16,device='cuda'),None,None]
        states=[sp_shard(v) if v is not None else None for v in full]
        current=starts.copy()
        for remain in range(19,0,-1):
            rows,current,_=decoder_tail_rows(current,remain,128)
            got=redistribute_sp_states(states,rows,total)
            mapping=mapping[rows]
            for i,v in got.items():
                expected=sp_shard(full[i].index_select(0,torch.as_tensor(mapping,device='cuda')))
                torch.testing.assert_close(v,expected,rtol=0,atol=0)
                states[i]=v
            total=len(rows)
        results.append({'lengths':lengths,'layers':19,'fields_exact':6})
    if rank==0:Path(f'/workspace/vllm_dsv41/artifacts/decoder-tail/exchange-tp{world}.json').write_text(json.dumps(results,indent=2)+'\n')
    destroy_model_parallel();destroy_distributed_environment()

if __name__=='__main__':
    for world in [2,4]:
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        mp.spawn(worker,args=(world,port),nprocs=world,join=True)
    print('PASS TP2/4, mixed rows, 19 successive redistributions, heterogeneous states')
