import json
from pathlib import Path
root=Path(__file__).parent
b=json.loads((root/'eval-dcp1.json').read_text())
summary={'baseline_correct':b['correct'],'n':32,'variants':{}}
for d in (2,4):
    path=root/f'eval-dcp{d}.json'
    if not path.exists():continue
    v=json.loads(path.read_text())
    if len(v['results'])!=32:continue
    assert v['truncated']==0
    assert v['long_retrieval']['tokens']==b['long_retrieval']['tokens']
    rows=[]
    exact=0;deltas=[]
    for a,c in zip(b['results'],v['results']):
        assert a['id']==c['id'] and a['prompt_len']==c['prompt_len']
        if a['tokens']==c['tokens']:exact+=1
        for ta,tc,la,lc in zip(a['tokens'],c['tokens'],a['logprobs'],c['logprobs']):
            if ta!=tc:
                ranked=sorted(la.items(),key=lambda x:x[1],reverse=True)
                rows.append({'id':a['id'],'baseline_token':ta,'candidate_token':tc,'baseline_top2_margin':ranked[0][1]-ranked[1][1] if len(ranked)>1 else None,'candidate_in_baseline_top5':str(tc) in la})
                break
            deltas.append(abs(la[str(ta)]-lc[str(tc)]))
    summary['variants'][d]={'correct':v['correct'],'exact_sequences':exact,
                            'same_final_answers':sum(a['answer']==c['answer'] for a,c in zip(b['results'],v['results'])),
                            'first_divergences':rows,'max_matched_prefix_logprob_delta':max(deltas,default=0),
                            'new_incorrect_ids':[a['id'] for a,c in zip(b['results'],v['results']) if a['correct'] and not c['correct']]}
(root/'eval-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
