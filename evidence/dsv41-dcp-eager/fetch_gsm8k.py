"""Fetch the pinned five-shot/32-question evaluation subset for local use."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = '3101c7d5072418e28b9008a6636bde82a006892c'
result = {'source_revision': REVISION, 'splits': {}, 'sha256': {}}
for split, count in [('train', 5), ('test', 32)]:
    url = f'https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}/grade_school_math/data/{split}.jsonl'
    data = urlopen(url, timeout=30).read()
    result['sha256'][split] = hashlib.sha256(data).hexdigest()
    result['splits'][split] = [json.loads(line) for line in data.decode().splitlines()][:count]
Path(__file__).with_name('gsm8k-subset.json').write_text(json.dumps(result, indent=2)+'\n')
print('Prepared five training demonstrations and 32 test questions at', REVISION)
