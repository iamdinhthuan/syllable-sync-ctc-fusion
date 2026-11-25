"""Equal-budget per-subset development retune of component CTC evidence (decode only).

Runs on the H100 worker. Standard library only; each decode is a child process of the
unchanged v2 decoder. Rule (frozen in PLAN.json before any decode):
  stage A: lambda sweep at the original (alpha, beta)
  stage B: beta in {b0-1, b0-0.5, b0+0.5, b0+1} and alpha in {a0-0.15, a0+0.15} at the stage-A lambda
  stage C: up to 4 bracketing rounds; the winner needs tested neighbours on both sides in
           lambda (adjacent grid value), alpha (+-0.15, except alpha=0) and beta (+-0.5).
Selection: minimum dev WER, then CER, |beta|, alpha, beta (the decoder's own rule).
All 15 lanes freeze before any test decode; each test runs exactly once.
"""
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/data/review_protocol/subset_retune_v1')
OUT = BASE / 'outputs'
CODE = Path('/data/review_protocol/v2/code')
PY = '/data/venv/bin/python'
PLAN = json.loads((BASE / 'PLAN.json').read_text())
ORIG = json.loads((BASE / 'original_subset_dev.json').read_text())
PER_POINT = ('ViMD',)  # AMENDMENT_1: per-point persistence for the slow corpus
CORE = ('decode_lm.py', 'decoding_protocol.py', 'nbest_rescore.py', 'replay_manifest.py',
        'train_vivos.py', 'tok.py', 'lm_model.py', 'ctc_prefix.py')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def log(*a):
    print(now(), *a, flush=True)


def key(l, a, b):
    return (round(l, 6), round(max(a, 0.0), 6), round(b, 6))


def fmt(p):
    return ':'.join(format(x, '.6g') for x in p)


def order(item):
    (l, a, b), m = item
    return (m['WER'], m['CER'], abs(b), a, b)


class Lane:
    def __init__(self, job, subset):
        self.job, self.subset = job, subset
        self.corpus = job['corpus']
        self.weights = PLAN['subsets'][subset]
        self.dir = OUT / self.corpus / subset
        self.points = {}      # key -> metrics
        self.where = {}       # key -> result file containing it
        self.rounds = 0
        l0, a0, b0 = map(float, job['full_parameters'].split(':'))
        self.l0, self.a0, self.b0 = l0, a0, b0

    def decode_dev(self, params, stage):
        params = [p for p in dict.fromkeys(params) if p not in self.points]
        if not params:
            return
        if self.corpus in PER_POINT:
            # one decoder call per grid point so that a lost worker loses at most one point
            for i, p in enumerate(params):
                self._decode(f'{stage}_p{i:02d}', [p])
            return
        self._decode(stage, params)

    def _decode(self, stage, params):
        out = self.dir / f'dev_{stage}'
        done = out / 'result_full.json'
        if done.exists():  # resume after worker loss: reuse a completed stage, never rerun it
            res = json.loads(done.read_text())
            got = {key(*map(float, k.split(':'))) for k in res['dev_grid']}
            if got != set(params):
                raise RuntimeError(f'{self.corpus}/{self.subset} {stage}: resumed grid differs')
            log(self.corpus, self.subset, stage, 'RESUMED completed stage')
            for k, m in res['dev_grid'].items():
                kk = key(*map(float, k.split(':')))
                self.points[kk] = m
                self.where[kk] = str(done)
            return
        if out.exists():  # incomplete stage from a lost worker: archive, then rerun the same grid
            out.rename(self.dir / f'incomplete_{stage}_{time.time_ns()}')
        cmd = [PY, str(CODE / 'decode_lm.py'), '--run', self.job['run'], '--root', self.job['root'],
               '--lm', self.job['lm'], '--max_sec', '40', '--out_dir', str(out), '--tag', 'full',
               '--dev_only', '--beam', '8', '--lex', 'lm', '--no_unk', '--comp_w', '1,1,1',
               '--ctc_cw', ','.join(map(str, self.weights)), '--grid', ','.join(fmt(p) for p in params)]
        log(self.corpus, self.subset, stage, 'points', len(params))
        t = time.time()
        with (self.dir / f'dev_{stage}.log').open('w') as lf:
            subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT, env=ENV)
        log(self.corpus, self.subset, stage, 'done', round(time.time() - t), 's')
        res = json.loads((out / 'result_full.json').read_text())
        for k, m in res['dev_grid'].items():
            kk = key(*map(float, k.split(':')))
            self.points[kk] = m
            self.where[kk] = str(out / 'result_full.json')

    def best(self):
        return min(self.points.items(), key=order)[0]

    def run_dev(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / 'FROZEN.json').exists():
            return json.loads((self.dir / 'FROZEN.json').read_text())
        lams = sorted(set(PLAN['lambda_grid']) | {self.l0})
        self.decode_dev([key(l, self.a0, self.b0) for l in lams], 'A')
        # reproduction check of the original fixed-setting point
        o = ORIG[f'{self.corpus}/{self.subset}']
        got = self.points[key(self.l0, self.a0, self.b0)]['WER']
        if abs(got - o['dev_WER']) > PLAN['reproduction_tolerance_wer']:
            raise RuntimeError(f'{self.corpus}/{self.subset}: original point dev WER {got} != {o["dev_WER"]}')
        l1 = self.best()[0]
        b_off, a_off = PLAN['beta_offsets'], PLAN['alpha_offsets']
        stageB = [key(l1, self.a0, self.b0 + d) for d in b_off]
        stageB += [key(l1, self.a0 + d, self.b0) for d in a_off]
        self.decode_dev(stageB, 'B')
        for r in range(PLAN['max_bracket_rounds']):
            l, a, b = self.best()
            need = []
            grid = sorted({k[0] for k in self.points} | set(lams))
            i = grid.index(l)
            lo = grid[i - 1] if i > 0 else round(l / 2, 6)
            hi = grid[i + 1] if i + 1 < len(grid) else round(min(PLAN['lambda_max'], (l + 1) / 2), 6)
            for ll in (lo, hi):
                if ll != l and 0 < ll <= PLAN['lambda_max']:
                    need.append(key(ll, a, b))
            for d in (-PLAN['beta_step'], PLAN['beta_step']):
                need.append(key(l, a, b + d))
            if a > 0:
                for d in (-PLAN['alpha_step'], PLAN['alpha_step']):
                    need.append(key(l, a + d, b))  # key() floors alpha at physical zero
            need = [p for p in need if p not in self.points]
            if not need:
                break
            self.rounds = r + 1
            self.decode_dev(need, f'C{r + 1}')
        l, a, b = self.best()
        bracketed = self.is_bracketed(l, a, b)
        frozen = dict(corpus=self.corpus, subset=self.subset, ctc_weights=self.weights,
                      best=fmt((l, a, b)), best_dev=self.points[(l, a, b)], frozen_config=self.where[(l, a, b)],
                      frozen_config_sha256=sha(self.where[(l, a, b)]), n_points=len(self.points),
                      bracket_rounds=self.rounds, bracketed=bracketed,
                      original=dict(setting=o['best'], dev_WER=o['dev_WER'], reproduced_dev_WER=got),
                      all_points={fmt(k): v for k, v in sorted(self.points.items())}, frozen_utc=now())
        (self.dir / 'FROZEN.json').write_text(json.dumps(frozen, indent=1))
        log(self.corpus, self.subset, 'FROZEN', frozen['best'], round(frozen['best_dev']['WER'], 3),
            'bracketed', bracketed)
        return frozen

    def is_bracketed(self, l, a, b):
        pts = self.points
        lam = any(k[0] < l and k[1:] == (a, b) for k in pts) and any(k[0] > l and k[1:] == (a, b) for k in pts)
        lam = lam or (l >= PLAN['lambda_max'])
        bet = any(k[2] < b and k[:2] == (l, a) for k in pts) and any(k[2] > b and k[:2] == (l, a) for k in pts)
        alp = a == 0 or (any(k[1] < a and (k[0], k[2]) == (l, b) for k in pts)
                         and any(k[1] > a and (k[0], k[2]) == (l, b) for k in pts))
        return dict(lam=lam, alpha=alp, beta=bet)

    def run_test(self, frozen):
        out = self.dir / 'test'
        if (out / 'result_full.json').exists():
            return
        if out.exists():
            out.rename(self.dir / f'incomplete_test_{time.time_ns()}')
        cmd = [PY, str(CODE / 'decode_lm.py'), '--run', self.job['run'], '--root', self.job['root'],
               '--lm', self.job['lm'], '--max_sec', '40', '--out_dir', str(out), '--tag', 'full',
               '--frozen_config', frozen['frozen_config'], '--export_nbest']
        log(self.corpus, self.subset, 'TEST start', frozen['best'])
        with (self.dir / 'test.log').open('w') as lf:
            subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT, env=ENV)
        r = json.loads((out / 'result_full.json').read_text())
        log(self.corpus, self.subset, 'TEST done', r['test'])


ENV = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HOME='/data/hf',
           HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')


def preflight():
    for c, o in ORIG.items():
        for name in CORE:
            if sha(CODE / name) != o['code_sha256'][name]:
                raise RuntimeError('frozen v2 source changed: ' + name)
    for job in PLAN['jobs']:
        o = ORIG[job['corpus'] + '/IRT']
        if sha(Path(job['run']) / 'best.pt') != o['checkpoint_sha256']:
            raise RuntimeError('checkpoint mismatch ' + job['corpus'])
        if sha(job['lm']) != o['lm_sha256']:
            raise RuntimeError('LM mismatch ' + job['corpus'])
    log('PREFLIGHT ok: 8 sources, 3 checkpoints, 3 LMs match recorded SHA-256')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preflight()
    lanes = [Lane(j, s) for j in PLAN['jobs'] for s in PLAN['subsets']]
    lanes.sort(key=lambda x: PLAN['lane_priority'].index(x.corpus))
    with cf.ThreadPoolExecutor(PLAN['concurrency']) as ex:
        frozen = list(ex.map(lambda x: x.run_dev(), lanes))
    allf = {f"{f['corpus']}/{f['subset']}": dict(best=f['best'], dev_WER=f['best_dev']['WER'],
            frozen_config_sha256=f['frozen_config_sha256'], bracketed=f['bracketed']) for f in frozen}
    if not (BASE / 'ALL_FROZEN.json').exists():
        (BASE / 'ALL_FROZEN.json').write_text(json.dumps(dict(frozen_utc=now(), lanes=allf), indent=1))
    log('ALL_FROZEN written; starting tests')
    with cf.ThreadPoolExecutor(PLAN['concurrency']) as ex:
        list(ex.map(lambda lf: lf[0].run_test(lf[1]), zip(lanes, frozen)))
    (BASE / 'DONE').write_text(now())
    log('DONE')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        (BASE / 'FAILED').write_text(now() + ' ' + repr(e))
        raise
