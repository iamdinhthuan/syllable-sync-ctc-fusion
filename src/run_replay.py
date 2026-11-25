"""Sequential inference only. At most one allocated CUDA device; no training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/data/review_protocol/v2')
CODE = BASE / 'code'
OUT = BASE / 'outputs'
PY = '/data/venv/bin/python'
SETTINGS = json.loads((CODE / 'REPLAY_SETTINGS.json').read_text())


def execute(command, directory, expected):
    directory.mkdir(parents=True, exist_ok=True)
    if expected.exists():
        json.loads(expected.read_text())
        print('ALREADY_RECORDED', expected, flush=True)
        return
    log = directory / ('run-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.log')
    print('START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), command, flush=True)
    with log.open('x') as stream:
        subprocess.run(command, check=True, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT)
    if not expected.is_file():
        raise RuntimeError(f'Command did not create {expected}; inspect {log}')
    print('FINISHED', expected, flush=True)


def decoder(job, directory, tag, extra):
    return [PY, str(CODE / 'decode_lm.py'), '--run', job['run'], '--root', job['root'],
            '--lm', job['lm'], '--max_sec', '40', '--out_dir', str(directory), '--tag', tag] + extra


def development(job, root, kind, limit=0):
    directory = root / ('dev_' + kind)
    extra = ['--dev_only', '--grid', job[kind + '_parameters'], '--beam', '8', '--lex', 'lm', '--no_unk']
    if limit:
        extra += ['--limit', str(limit)]
    record = directory / ('result_' + kind + '.json')
    execute(decoder(job, directory, kind, extra), directory, record)
    return record


def tests(job, root, lm_config, full_config, limit=0, external=False):
    for kind, config in [('lm', lm_config), ('full', full_config)]:
        directory = root / ('test_' + kind)
        extra = ['--frozen_config', str(config)]
        if kind == 'lm':
            extra += ['--export_nbest']
        if limit:
            extra += ['--limit', str(limit)]
        if external:
            extra += ['--eval_root', str(BASE / 'fleurs'), '--evaluation_manifest', str(CODE / 'EXTERNAL_EVAL_PRECOMMIT.json')]
        execute(decoder(job, directory, kind, extra), directory, directory / ('result_' + kind + '.json'))
    directory = root / 'rescored'
    command = [PY, str(CODE / 'rescore_saved_nbest.py'),
               '--nbest', str(root / 'test_lm' / 'nbest_lm.json'),
               '--frozen_config', str(full_config),
               '--references', str(root / 'test_lm' / 'hyps_lm.json'),
               '--out', str(directory)]
    # The sidecar creates its own exclusive output directory; log alongside it.
    execute(command, root / 'cpu_rescoring_logs', directory / 'result.json')


def main():
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Inference protocol requires exactly one visible CUDA device')
    OUT.mkdir(parents=True, exist_ok=True)
    print('INFERENCE_ONLY', torch.cuda.get_device_name(0), flush=True)
    smoke = SETTINGS['jobs'][0]
    root = OUT / 'smoke'
    lm = development(smoke, root, 'lm', 2)
    full = development(smoke, root, 'full', 2)
    tests(smoke, root, lm, full, 2)
    print('GPU_SMOKE_PASSED', flush=True)
    configs = {}
    for job in SETTINGS['jobs']:
        root = OUT / job['corpus']
        configs[job['corpus']] = (
            development(job, root, 'lm'), development(job, root, 'full'))
        tests(job, root, *configs[job['corpus']])
    vimd = next(job for job in SETTINGS['jobs'] if job['corpus'] == 'ViMD')
    tests(vimd, OUT / 'external' / 'FLEURS', *configs['ViMD'], external=True)
    (OUT / 'ALL_DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
