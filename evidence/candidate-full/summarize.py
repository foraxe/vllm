import json
from pathlib import Path
from statistics import median
root=Path(__file__).parent
runs={label:json.loads((root/f'model-{label}.json').read_text()) for label in ('a1','b1','b2','a2')}
ref=runs['a1']
for label,run in runs.items():
    assert len(run['results'])==18
    assert run['mixed_outputs']==ref['mixed_outputs']
    assert run['diagnostic_output']==ref['diagnostic_output']
    for a,b in zip(ref['results'],run['results']):
        assert a['token_ids']==b['token_ids'],(label,b['context'],b['trial'])
        assert b['steps']['prefill']==(b['context']+8191)//8192 and b['steps']['decode']==63
summary={'runner':'MRV2','exact_tokens':True,'matched_steps':True,'contexts':{}}
for n in (17,8192,32768):
    row={'runs':{}}
    for label,run in runs.items():
        data=[x for x in run['results'] if x['context']==n and not x['warmup']]
        row['runs'][label]={k:median(x[k] for x in data) for k in ('ttft_ms','prefill_ms','tpot_ms')}
    for arm in ('a','b'):
        data=[x for label,run in runs.items() if label.startswith(arm) for x in run['results'] if x['context']==n and not x['warmup']]
        row[arm]={k:median(x[k] for x in data) for k in ('ttft_ms','prefill_ms','tpot_ms')}
    row['change_pct']={k:100*(row['b'][k]/row['a'][k]-1) for k in row['a']}
    row['both_orders_favor_candidate']=all(row['runs']['b'+i]['ttft_ms']<row['runs']['a'+i]['ttft_ms'] for i in ('1','2'))
    summary['contexts'][n]=row
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
