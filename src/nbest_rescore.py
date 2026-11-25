"""Exact completed-component CTC rescoring of an already exported LM N-best list."""
import math
from numbers import Integral

import torch
import torch.nn.functional as F


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _number(value, name, allow_negative_infinity=False):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) and not (allow_negative_infinity and value == -math.inf):
        raise ValueError(f"{name} must be finite" +
                         (" or negative infinity" if allow_negative_infinity else ""))
    return value


@torch.no_grad()
def rescore_component_nbest(candidates, log_probs, input_length, ctc_weight,
                            lm_weight, insertion_bonus, component_weights=None, blank=0):
    """Return copied, completed records sorted by descending full-sequence score.

    ``log_probs`` contains one normalized Tensor[T, Vc] per component; only
    ``[:input_length]`` is scored. Each candidate has nonterminal ``component_ids``
    (one C-tuple per syllable), ``attention_logprob`` and ``lm_logprob`` already
    including natural EOS, ``length_bonus_count``, and a boolean ``ended``.
    Additional candidate fields are preserved. Inputs are not modified.

    Records with ended=False are explicitly excluded, without appending EOS or
    scoring their partial transcript. The caller must retain/count those records
    separately for coverage and termination reporting. All-unfinished input
    returns an empty list; there is no fallback to a partial candidate.

    score = (1-lambda)*attention + lambda*sum(wc*CTC_c) + alpha*LM + beta*count.
    Lambda is in [0,1]; alpha and component weights are nonnegative; beta may be
    negative. All weights are finite. Scores may be -inf, but not NaN or +inf.
    Zero-coefficient terms are omitted. Exact completed CTC log probabilities
    are still reported for every component, including impossible targets (-inf).
    Default component weights are ones, without normalization or retuning.

    CTC uses float64 on the input device, sum reduction and zero_infinity=False;
    no model loading, search, transcript reconstruction, or CLI is performed.
    """
    ctc_weight = _number(ctc_weight, 'ctc_weight')
    lm_weight = _number(lm_weight, 'lm_weight')
    insertion_bonus = _number(insertion_bonus, 'insertion_bonus')
    if not 0 <= ctc_weight <= 1 or lm_weight < 0:
        raise ValueError('ctc_weight must be in [0,1] and lm_weight nonnegative')
    input_length = _integer(input_length, 'input_length')
    blank = _integer(blank, 'blank')
    if input_length < 0 or not log_probs:
        raise ValueError('input_length must be nonnegative and log_probs nonempty')
    count = len(log_probs)
    weights = [1.0] * count if component_weights is None else list(component_weights)
    if len(weights) != count:
        raise ValueError('component_weights must have one entry per component')
    weights = [_number(w, 'component weight') for w in weights]
    if any(w < 0 for w in weights):
        raise ValueError('component weights must be nonnegative')

    streams = []
    for lp in log_probs:
        if not isinstance(lp, torch.Tensor) or lp.ndim != 2 or not lp.is_floating_point():
            raise ValueError('each log_probs entry must be a floating Tensor[T,Vc]')
        if input_length > lp.shape[0] or not 0 <= blank < lp.shape[1]:
            raise ValueError('input_length or blank is outside a component tensor')
        lp = lp[:input_length].detach().to(dtype=torch.float64)
        if torch.isnan(lp).any() or torch.isposinf(lp).any():
            raise ValueError('valid-frame log_probs cannot contain NaN or +inf')
        streams.append(lp)

    output = []
    for record in candidates:
        if not isinstance(record.get('ended'), bool):
            raise ValueError('each candidate must have a boolean ended flag')
        if not record['ended']:
            continue
        attention = _number(record['attention_logprob'], 'attention_logprob', True)
        lm = _number(record['lm_logprob'], 'lm_logprob', True)
        bonus_count = _integer(record['length_bonus_count'], 'length_bonus_count')
        if bonus_count < 0:
            raise ValueError('length_bonus_count must be nonnegative')
        targets = [[] for _ in streams]
        for ids in record['component_ids']:
            if len(ids) != count:
                raise ValueError('each component_ids row must contain C labels')
            for j, label in enumerate(ids):
                label = _integer(label, 'component label')
                if not 0 <= label < streams[j].shape[1] or label == blank:
                    raise ValueError('component targets must be in-range nonblank labels')
                targets[j].append(label)

        ctc = []
        for lp, labels in zip(streams, targets):
            minimum_frames = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
            if not labels:
                value = float(lp[:, blank].sum())
            elif minimum_frames > input_length:
                value = -math.inf
            else:
                target = torch.tensor(labels, dtype=torch.long, device=lp.device)
                loss = F.ctc_loss(lp[:, None, :], target, (input_length,), (len(labels),),
                                  blank=blank, reduction='sum', zero_infinity=False)
                value = -float(loss)
            ctc.append(value)

        terms = [(1 - ctc_weight, attention), (lm_weight, lm)]
        terms.extend((ctc_weight * w, value) for w, value in zip(weights, ctc))
        score = sum(w * value for w, value in terms if w != 0)
        score += insertion_bonus * bonus_count
        if math.isnan(score) or score == math.inf:
            raise ValueError('rescored_score overflowed to NaN or +inf')
        output.append(dict(record, ctc_logprobs=ctc, rescored_score=score))
    return sorted(output, key=lambda row: row['rescored_score'], reverse=True)
