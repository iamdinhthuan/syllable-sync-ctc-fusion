"""Small exhaustive CPU checks of the frozen ComponentCTC, without CTC recurrences.

Run: python -B test_ctc_prefix_exhaustive.py -v
Override the source file/directory with an absolute CTC_PREFIX_SOURCE path.
The default is the sibling paper_csl/review_revision/protocol_v2/ctc_prefix.py.
"""

import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
import unittest
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
import torch


SCORER_SHA256 = "a50d8398f596042d7358195b2ce3982badb1916f09190b5b0f7873d453c4d008"
PROBABILITY_ATOL = 1e-12
POSITIVE_LOG_ATOL = 1e-10
VALID_LENGTHS = (2, 3, 4)


def load_component():
    default = (Path(__file__).resolve().parents[2] / "paper_csl" /
               "review_revision/protocol_v2/ctc_prefix.py")
    source = Path(os.environ.get("CTC_PREFIX_SOURCE", str(default))).expanduser()
    if not source.is_absolute():
        raise ValueError("CTC_PREFIX_SOURCE must be an absolute file or directory")
    if source.is_dir():
        source = source / "ctc_prefix.py"
    source = source.resolve(strict=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SCORER_SHA256:
        raise ValueError(f"Scorer is not the frozen version: {source}, SHA256={digest}")
    spec = importlib.util.spec_from_file_location("archived_frozen_ctc_prefix", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ComponentCTC, source


def collapse(path):
    # Merge first, remove blank second: (1, 0, 1) must remain (1, 1).
    return tuple(label for label, _ in itertools.groupby(path) if label != 0)


def enumerate_oracle(logp, valid_length, counts):
    """Direct path sums; padded horizons have deterministic trailing blanks."""
    probabilities = logp.exp().tolist()
    frames, alphabet = logp.shape
    prefix_mass, complete = defaultdict(float), defaultdict(float)
    for path in itertools.product(range(alphabet), repeat=valid_length):
        mass = math.prod(probabilities[t][label] for t, label in enumerate(path))
        output = collapse(path)
        complete[output] += mass
        for length in range(len(output) + 1):
            prefix_mass[output[:length]] += mass
        counts["full_paths_enumerated"] += 1

    states = []
    for horizon in range(1, frames + 1):
        nonblank, blank = defaultdict(float), defaultdict(float)
        for path in itertools.product(range(alphabet), repeat=min(horizon, valid_length)):
            mass = math.prod(probabilities[t][label] for t, label in enumerate(path))
            ends_blank = horizon > valid_length or path[-1] == 0
            (blank if ends_blank else nonblank)[collapse(path)] += mass
            counts["forward_paths_enumerated"] += 1
        states.append((nonblank, blank))
    return dict(prefix=prefix_mass, complete=complete, states=states)


def emission_batch(alphabet):
    # Different row/frame distributions expose missing masking and wrong parents.
    logits = torch.tensor([
        [[(((row + 2) * (frame + 3) * (label + 1)) % 13 - 6) / 3
          for label in range(alphabet)] for frame in range(4)]
        for row in range(len(VALID_LENGTHS))
    ], dtype=torch.float64, device="cpu")
    return logits.log_softmax(-1)


class ExhaustiveComponentCTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.component, cls.source = load_component()
        cls.counts = Counter()
        cls.max_probability_error = 0.0
        cls.max_positive_log_error = 0.0
        cls.fixtures = {}
        for alphabet in (2, 3):
            logp = emission_batch(alphabet)
            assert bool(torch.isfinite(logp).all())
            oracles = [enumerate_oracle(logp[row], length, cls.counts)
                       for row, length in enumerate(VALID_LENGTHS)]
            cls.fixtures[alphabet] = logp, oracles

    @classmethod
    def tearDownClass(cls):
        print("EXHAUSTIVE_CTC_SCOPE " + json.dumps(dict(
            source=str(cls.source), source_sha256=SCORER_SHA256,
            probability_atol=PROBABILITY_ATOL, positive_log_atol=POSITIVE_LOG_ATOL,
            counts=dict(sorted(cls.counts.items())),
            maximum_probability_error=cls.max_probability_error,
            maximum_positive_log_error=cls.max_positive_log_error,
            torch=torch.__version__, python=sys.version.split()[0], device="cpu",
        ), sort_keys=True))

    def check_probability(self, actual_log, expected, where):
        actual_log = float(actual_log)
        actual = math.exp(actual_log)
        self.assertTrue(math.isfinite(actual), where)
        cls = type(self)
        cls.max_probability_error = max(cls.max_probability_error, abs(actual - expected))
        self.counts["probability_comparisons"] += 1
        self.assertAlmostEqual(actual, expected, delta=PROBABILITY_ATOL, msg=where)
        if expected > 0:
            error = abs(actual_log - math.log(expected))
            cls.max_positive_log_error = max(cls.max_positive_log_error, error)
            self.counts["positive_log_comparisons"] += 1
            self.assertLessEqual(error, POSITIVE_LOG_ATOL, where)
        # An oracle zero does not require the finite-sentinel code to return -inf.

    def check_state(self, state, prefixes, origins, oracles, kind):
        next_prefix, completion = state.scores()
        self.assertEqual(state.x.device.type, "cpu")
        self.assertEqual(state.L, len(prefixes[0]))
        for row, (prefix, origin) in enumerate(zip(prefixes, origins)):
            oracle = oracles[origin]
            context = f"{kind}, row={row}, original={origin}, prefix={prefix}"
            self.counts[kind + "_prefix_rows"] += 1
            self.assertEqual(int(state.last[row]), prefix[-1] if prefix else -1)
            self.check_probability(state.psi[row], oracle["prefix"].get(prefix, 0), context + " Psi")
            self.check_probability(completion[row], oracle["complete"].get(prefix, 0), context + " P")
            for frame, (nonblank, blank) in enumerate(oracle["states"]):
                self.check_probability(state.rn[row, frame], nonblank.get(prefix, 0), context + f" rn@{frame + 1}")
                self.check_probability(state.rb[row, frame], blank.get(prefix, 0), context + f" rb@{frame + 1}")
            for label in range(1, state.x.shape[1]):
                extended = prefix + (label,)
                self.counts[kind + "_extension_rows"] += 1
                self.check_probability(next_prefix[row, label], oracle["prefix"].get(extended, 0), context + f" Psi+{label}")
                for frame, (nonblank, blank) in enumerate(oracle["states"]):
                    self.check_probability(state._rn_h[row, label, frame], nonblank.get(extended, 0), context + f" rn+{label}@{frame + 1}")
                    self.check_probability(state._rb_h[row, label, frame], blank.get(extended, 0), context + f" rb+{label}@{frame + 1}")

    def test_exhaustive_nonterminal_prefix_mass_and_forward_states(self):
        # Fails for completion substituted for Psi, wrong repeats or padding masks.
        lengths = torch.tensor(VALID_LENGTHS, device="cpu")
        identity = torch.arange(len(VALID_LENGTHS), device="cpu")
        for alphabet, (logp, oracles) in self.fixtures.items():
            for length in range(5):
                for prefix in itertools.product(range(1, alphabet), repeat=length):
                    with self.subTest(alphabet=alphabet, prefix=prefix):
                        state = self.component(logp, lengths, blank=0)
                        for label in prefix:
                            state.scores()
                            state.select(identity, torch.full_like(identity, label))
                        self.check_state(state, [prefix] * 3, [0, 1, 2], oracles, "exhaustive")
        self.assertEqual(self.counts["exhaustive_prefix_rows"], 108)
        self.assertEqual(self.counts["exhaustive_extension_rows"], 201)

    def test_oracle_hand_counts_distinguish_prefix_completion_and_blank(self):
        self.assertEqual(collapse((1, 1, 0, 1)), (1, 1))
        self.assertEqual(collapse((0, 0)), ())
        logp = torch.zeros(3, 3, dtype=torch.float64, device="cpu").log_softmax(-1)
        two = enumerate_oracle(logp[:2], 2, self.counts)
        self.assertAlmostEqual(two["prefix"][(1,)], 4 / 9, delta=1e-14)
        self.assertAlmostEqual(two["complete"][(1,)], 1 / 3, delta=1e-14)
        self.assertAlmostEqual(two["complete"][()], 1 / 9, delta=1e-14)
        self.assertAlmostEqual(two["states"][1][0][(1,)], 2 / 9, delta=1e-14)
        self.assertAlmostEqual(two["states"][1][1][(1,)], 1 / 9, delta=1e-14)
        self.assertEqual(two["prefix"].get((1, 1), 0), 0)
        three = enumerate_oracle(logp, 3, self.counts)
        self.assertAlmostEqual(three["complete"][(1, 1)], 1 / 27, delta=1e-14)

    def test_parent_selection_reorders_duplicates_and_shrinks_rows(self):
        # Fails if select() reorders histories but leaves old emission rows/states.
        logp, oracles = self.fixtures[3]
        state = self.component(logp, torch.tensor(VALID_LENGTHS, device="cpu"), blank=0)
        prefixes, origins = [()] * 3, [0, 1, 2]
        self.check_state(state, prefixes, origins, oracles, "selection")
        for parents, labels in [([2, 0, 2, 1], [1, 1, 2, 2]),
                                ([2, 3, 0, 2], [2, 1, 1, 2]),
                                ([1, 2, 0], [1, 2, 2])]:
            state.select(torch.tensor(parents, device="cpu"), torch.tensor(labels, device="cpu"))
            prefixes = [prefixes[p] + (label,) for p, label in zip(parents, labels)]
            origins = [origins[p] for p in parents]
            self.check_state(state, prefixes, origins, oracles, "selection")
        self.assertEqual(self.counts["selection_prefix_rows"], 14)
        self.assertEqual(self.counts["selection_extension_rows"], 28)


if __name__ == "__main__":
    unittest.main()
