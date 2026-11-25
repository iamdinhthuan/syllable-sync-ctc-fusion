"""CPU-only exact-CTC reranking of saved baseline-LM N-best lists; no models."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from rapidfuzz.distance.Levenshtein import distance


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite JSON number')
    return float(value)


def _log_number(value, name):
    if value == '-inf':
        return -math.inf
    return _finite(value, name)


def _weights(pool, config):
    if config.get('selection_scope') != 'development':
        raise ValueError('frozen_config selection_scope must be development')
    try:
        values = config['best'].split(':')
        if len(values) != 3:
            raise ValueError
        lam, alpha, beta = [_finite(float(v), 'best weight') for v in values]
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError('frozen_config best must be finite lambda:alpha:beta') from exc
    if not 0 <= lam <= 1 or alpha < 0:
        raise ValueError('lambda must be in [0,1] and alpha nonnegative')
    protocols = [pool.get('protocol'), config.get('protocol')]
    if not all(isinstance(p, dict) for p in protocols):
        raise ValueError('both inputs require protocol objects')
    for key in ['checkpoint_sha256', 'lm_sha256', 'inventory_sha256']:
        hashes = [p.get(key) for p in protocols]
        if not all(isinstance(h, str) and re.fullmatch(r'[0-9a-fA-F]{64}', h) for h in hashes):
            raise ValueError(f'missing or invalid {key}')
        if hashes[0].lower() != hashes[1].lower():
            raise ValueError(f'protocol mismatch: {key}')
    component_vectors = {}
    for key in ['attention_weights', 'ctc_weights']:
        vectors = []
        for protocol in protocols:
            vector = protocol.get(key)
            if not isinstance(vector, list) or not vector:
                raise ValueError(f'{key} must be a nonempty component-weight list')
            vector = [_finite(v, key) for v in vector]
            if any(v < 0 for v in vector):
                raise ValueError(f'{key} must be nonnegative')
            vectors.append(vector)
        if vectors[0] != vectors[1]:
            raise ValueError(f'protocol mismatch: {key}')
        component_vectors[key] = vectors[0]
    if len(component_vectors['attention_weights']) != len(component_vectors['ctc_weights']):
        raise ValueError('attention_weights and ctc_weights must have equal lengths')
    for key in ['beam', 'lex', 'no_unk']:
        if any(key in p for p in protocols):
            if not all(key in p for p in protocols) or protocols[0][key] != protocols[1][key]:
                raise ValueError(f'protocol mismatch: {key}')
    return lam, alpha, beta, component_vectors['ctc_weights']


def _rank(rows, weights):
    """Select without references; search_score never enters the new objective."""
    if not isinstance(rows, list):
        raise ValueError('each utterance must contain a candidate list')
    counts = dict(candidates=len(rows), ended_candidates=0, unfinished_candidates=0,
                  finite_complete_candidates=0, nonfinite_complete_candidates=0,
                  unscored_complete_candidates=0)
    best = None
    lam, alpha, beta, cw = weights
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get('hyp'), str):
            raise ValueError('each candidate requires a string hyp')
        if not isinstance(row.get('ended'), bool):
            raise ValueError('each candidate requires a boolean ended')
        if not row['ended']:
            counts['unfinished_candidates'] += 1
            continue
        counts['ended_candidates'] += 1
        required = {'attention_logprob', 'lm_logprob', 'ctc_logprobs', 'length_bonus_count'}
        if not required <= row.keys():
            counts['unscored_complete_candidates'] += 1
            continue
        values = row['ctc_logprobs']
        if not isinstance(values, list) or len(values) != len(cw):
            raise ValueError('ctc_logprobs must have one exact score per component')
        ctc = [_log_number(v, 'ctc_logprobs') for v in values]
        attention = _log_number(row['attention_logprob'], 'attention_logprob')
        lm = _log_number(row['lm_logprob'], 'lm_logprob')
        count = row['length_bonus_count']
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('length_bonus_count must be a nonnegative integer')
        terms = [(1 - lam, attention), (alpha, lm)]
        terms.extend((lam * w, v) for w, v in zip(cw, ctc))
        score = sum(w * v for w, v in terms if w != 0) + beta * count
        if not math.isfinite(score):
            counts['nonfinite_complete_candidates'] += 1
            continue
        counts['finite_complete_candidates'] += 1
        if best is None or score > best[1]['rescored_score']:
            best = (index, dict(row, rescored_score=score))
    return best, counts


def _rates(word_errors, tokens, char_errors, characters, utterances):
    return dict(WER=100 * word_errors / tokens if tokens else None,
                CER=100 * char_errors / characters if characters else None,
                word_errors=word_errors, reference_tokens=tokens,
                char_errors=char_errors, reference_characters=characters,
                utterances=utterances)


def _metrics(references, hypotheses):
    we = n = ce = nc = 0
    for uid, row in references.items():
        ref, hyp = row['ref'], hypotheses[uid]
        we += distance(ref.split(), hyp.split())
        ce += distance(ref, hyp)
        n += len(ref.split())
        nc += len(ref)
    return _rates(we, n, ce, nc, len(references))


def _oracle(references, utterances, complete_only):
    we = n = ce = nc = covered = 0
    for uid, row in references.items():
        ref = row['ref']
        choices = [c['hyp'] for c in utterances.get(uid, []) if c['ended'] or not complete_only]
        if choices:
            covered += 1
        else:
            choices = [row['hyp']]
        we += min(distance(ref.split(), hyp.split()) for hyp in choices)
        ce += min(distance(ref, hyp) for hyp in choices)
        n += len(ref.split())
        nc += len(ref)
    return dict(_rates(we, n, ce, nc, len(references)), coverage_utterances=covered,
                fallback_utterances=len(references) - covered,
                empty_pool_policy='baseline_fallback', metric_minima_are_independent=True)


def rescore_saved_nbest(pool, frozen_config, references):
    """Return protocol, hyps, selections, metrics, coverage and oracle diagnostics.

    Inputs are parsed JSON dictionaries. Natural EOS is already in the saved
    attention/LM totals; CTC values are exact completed probabilities supplied by
    the exporter. No logits, model loading, search, or transcript changes occur.
    Only finite full-objective scores of ended, scored candidates are eligible.
    Zero-weight -inf terms are ignored, consistently with rescore_component_nbest.
    Every baseline UID survives; unusable lists fall back to its original hyp.
    References are first read after all candidate rankings and system choices.
    Oracles use independent WER/CER minima and explicit baseline fallback where
    their complete/any candidate pool is empty, keeping the full denominator.
    """
    if not all(isinstance(value, dict) for value in [pool, frozen_config, references]):
        raise ValueError('pool, frozen_config and references must be JSON objects')
    weights = _weights(pool, frozen_config)
    utterances = pool.get('utterances')
    if not isinstance(utterances, dict):
        raise ValueError('pool requires an utterances object')
    ranked = {uid: _rank(rows, weights) for uid, rows in utterances.items()}
    if set(utterances) - set(references):
        raise ValueError('pool contains UIDs absent from the baseline references file')
    coverage = Counter(dict(utterances=len(references), candidates=0, ended_candidates=0,
        unfinished_candidates=0, finite_complete_candidates=0, nonfinite_complete_candidates=0,
        unscored_complete_candidates=0, rescored_utterances=0, fallback_utterances=0,
        cap_fallback_utterances=0, unscored_fallback_utterances=0,
        nonfinite_fallback_utterances=0, missing_pool_fallback_utterances=0,
        empty_pool_fallback_utterances=0))
    hypotheses, selections = {}, {}
    for uid, baseline in references.items():
        if not isinstance(baseline, dict) or not isinstance(baseline.get('hyp'), str):
            raise ValueError('each baseline entry requires its original string hyp')
        best, counts = ranked.get(uid, (None, dict(candidates=0, ended_candidates=0,
            unfinished_candidates=0, finite_complete_candidates=0,
            nonfinite_complete_candidates=0, unscored_complete_candidates=0)))
        coverage.update(counts)
        reasons = []
        if best is None:
            coverage['fallback_utterances'] += 1
            if uid not in utterances:
                reasons.append('missing_pool')
            elif not utterances[uid]:
                reasons.append('empty_pool')
            else:
                for reason, key in [('cap', 'unfinished_candidates'),
                                    ('unscored', 'unscored_complete_candidates'),
                                    ('nonfinite', 'nonfinite_complete_candidates')]:
                    if counts[key]:
                        reasons.append(reason)
            for reason in reasons:
                coverage[reason + '_fallback_utterances'] += 1
            hypotheses[uid] = baseline['hyp']
        else:
            coverage['rescored_utterances'] += 1
            hypotheses[uid] = best[1]['hyp']
        selections[uid] = dict(source='nbest' if best else 'baseline_fallback',
            hyp=hypotheses[uid], candidate_index=best[0] if best else None,
            candidate=best[1] if best else None, fallback_reasons=reasons, counts=counts)

    # Reference content is accessed only after selection, including every fallback.
    if any(not isinstance(row.get('ref'), str) for row in references.values()):
        raise ValueError('each reference entry requires a string ref')
    lam, alpha, beta, cw = weights
    result = dict(protocol=dict(method='saved_baseline_lm_nbest_completed_ctc',
        selection_uses_references=False, selection_scope=frozen_config['selection_scope'],
        best=frozen_config['best'], target_weights=dict(ctc_weight=lam, lm_weight=alpha,
            insertion_bonus=beta, component_weights=cw),
        pool_protocol=pool['protocol'], frozen_protocol=frozen_config['protocol']),
        hyps={uid: dict(ref=row['ref'], hyp=hypotheses[uid]) for uid, row in references.items()},
        selections=selections, metrics=_metrics(references, hypotheses), coverage=dict(coverage),
        oracle=dict(complete=_oracle(references, utterances, True),
                    any=_oracle(references, utterances, False)))
    json.dumps(result, allow_nan=False)
    return result


def _reject_constant(value):
    raise ValueError(f'nonstandard JSON constant {value}; encode negative infinity as "-inf"')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['nbest', 'frozen_config', 'references', 'out']:
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        documents, sources = {}, {}
        for key in ['nbest', 'frozen_config', 'references']:
            path = getattr(args, key)
            raw = path.read_bytes()
            documents[key] = json.loads(raw, parse_constant=_reject_constant,
                                        object_pairs_hook=_unique_object)
            sources[key] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        result = rescore_saved_nbest(documents['nbest'], documents['frozen_config'],
                                     documents['references'])
        hyps = result.pop('hyps')
        result['protocol']['input_files'] = sources
        outputs = {'hyps.json': hyps, 'result.json': result}
        texts = {name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
                 for name, value in outputs.items()}
        args.out.mkdir(parents=True, exist_ok=False)
        for name, text in texts.items():
            with (args.out / name).open('x', encoding='utf-8') as stream:
                stream.write(text)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(f'Saved-list rescoring complete: {args.out}; '
          f"fallbacks={result['coverage']['fallback_utterances']}")


if __name__ == '__main__':
    main()
