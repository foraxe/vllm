import json,os,subprocess
from pathlib import Path
root=Path('/workspace/vllm_dsv41');out=root/'artifacts/candidate-full'
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',VLLM_USE_V2_MODEL_RUNNER='1',VLLM_DEEP_GEMM_WARMUP='skip',PYTHONPATH=f'{out}:{root}/runtime-deps/nvidia_cutlass_dsl/dsl_packages:{root}/runtime-deps:{root}/vllm-fused-out')
ref=json.loads((out/'model-a1.json').read_text())
assert len(ref['results'])==18
for label,variant in [('b1','candidate'),('b2','candidate'),('a2','baseline')]:
    print(f'START {label}',flush=True)
    dest=out/f'model-{label}.json'
    with (out/f'model-{label}.log').open('w') as log:
        subprocess.run([str(root/'vllm-fused-out/.venv/bin/python'),str(out/'candidate_model_bench.py'),'--variant',variant,'--contexts','17','8192','32768','--trials','4','--output',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    data=json.loads(dest.read_text())
    assert data['mixed_outputs']==ref['mixed_outputs'],label
    assert data['diagnostic_output']==ref['diagnostic_output'],label
    assert len(data['results'])==18
    for a,b in zip(ref['results'],data['results']):
        assert a['token_ids']==b['token_ids'],(label,b['context'],b['trial'])
        for key in ('prefill','decode'):
            assert a['steps'][key]==b['steps'][key],(label,b)
    print(f'DONE {label}: exact tokens and matched steps',flush=True)
