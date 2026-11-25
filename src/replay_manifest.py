"""Small provenance helpers; frozen replay does not erase historical test exposure."""
import hashlib
import json
import math
from pathlib import Path


def file_sha256(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def parse_parameters(text):
    values=tuple(float(x) for x in text.split(':'))
    if len(values)==2:values=(0.,*values)
    if len(values)!=3 or not all(math.isfinite(x) for x in values):
        raise ValueError('Expected finite lambda:alpha:beta (or alpha:beta)')
    if not 0<=values[0]<=1 or values[1]<0:
        raise ValueError('CTC weight must be in [0,1] and LM weight nonnegative')
    return values


def load_frozen_config(path):
    record=json.loads(Path(path).read_text())
    if record.get('selection_scope')!='development' or not isinstance(record.get('protocol'),dict):
        raise ValueError('Frozen settings must have an explicit development-selection record')
    parse_parameters(record['best'])
    return record


def verify_resources(saved,current):
    keys=['checkpoint_sha256','lm_sha256','inventory_sha256','attention_weights','ctc_weights']
    if 'code_sha256' in saved:keys.append('code_sha256')
    for key in keys:
        if key not in saved or saved[key]!=current.get(key):
            raise ValueError(f'Frozen configuration resource mismatch: {key}')


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        if value==-math.inf:return '-inf'
        raise ValueError('NaN and positive infinity are not valid evaluation artifacts')
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(x) for x in value]
    return value


def dump_record(path,record):
    text=json.dumps(json_safe(record),ensure_ascii=False,indent=1,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(text)


def load_replay_splits(loader,eval_root='',dev_only=False):
    training_root=loader.ROOT
    train=loader.load_split('train')
    if dev_only:return train,[]
    try:
        loader.ROOT=eval_root or training_root
        test=loader.load_split('test')
    finally:loader.ROOT=training_root
    return train,test
