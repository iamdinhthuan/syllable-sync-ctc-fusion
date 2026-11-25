"""Factorized CTC prefix scoring for syllable-synchronous beam search.

Each syllable component (initial, rhyme, tone) has its own frame-level CTC head. For a hypothesis prefix g (a sequence of
syllables) and a candidate syllable y = (y_1..y_C), the joint CTC prefix score is the sum over components of the CTC prefix
log-probability of g_c + y_c (Watanabe et al., 2017), computed for every label of every component at once.

The time recursion is evaluated with cumulative sums and logcumsumexp instead of a Python loop over frames:
    r^n_t(h) = logaddexp(r^n_{t-1}(h), phi_{t-1}) + x_t(c)   ->  r^n_t = S_t + logcumsumexp_tau(phi_{tau-1} - S_{tau-1})
    r^b_t(h) = logaddexp(r^n_{t-1}(h), r^b_{t-1}(h)) + x_t(blank) -> r^b_t = B_t + logcumsumexp_{tau<t}(r^n_tau - B_tau)
    psi(h)   = logsumexp_t (phi_{t-1} + x_t(c))
with S, B the cumulative log-probabilities of label c and of blank.
"""
import torch

NEG = -1e30          # stands for log(0) in the states
PAD_LOGP = -1e4      # log-prob of a non-blank label on padded frames (exp(-1e4) == 0, but stays finite for the cumsums)


def _lcse_excl(z):
    """exclusive logcumsumexp along the last dim: out[t] = logsumexp(z[..., :t]), out[0] = -inf."""
    c = torch.logcumsumexp(z, -1)
    return torch.cat([torch.full_like(c[..., :1], NEG), c[..., :-1]], -1)


class ComponentCTC:
    """CTC prefix state for one component over a batch of hypotheses (rows)."""

    def __init__(self, logp, lens, blank=0):
        # logp: (R, T, V) log-softmax of the CTC head, lens: (R,) valid frames per row
        R, T, V = logp.shape
        x = logp.double().clone()
        pad = torch.arange(T, device=x.device)[None, :] >= lens[:, None]           # (R, T)
        x[pad] = PAD_LOGP
        x[..., blank] = torch.where(pad, torch.zeros_like(x[..., blank]), x[..., blank])
        self.x = x.transpose(1, 2).contiguous()                                      # (R, V, T)
        self.xb = x[..., blank]                                                      # (R, T)
        self.S = torch.cumsum(self.x, -1)                                            # (R, V, T)
        self.B = torch.cumsum(self.xb, -1)                                           # (R, T)
        self.blank, self.T = blank, T
        # empty prefix: r^n = -inf, r^b_t = sum of blank log-probs up to t
        self.rn = torch.full((R, T), NEG, dtype=torch.float64, device=x.device)
        self.rb = self.B.clone()
        self.psi = torch.zeros(R, dtype=torch.float64, device=x.device)
        self.last = torch.full((R,), -1, dtype=torch.long, device=x.device)
        self.L = 0

    def scores(self):
        """psi for appending every label (R, V), plus log P(prefix) for ending (R,). Keeps per-label states for select()."""
        rsum = torch.logaddexp(self.rn, self.rb)                                      # (R, T)
        R, V, T = self.x.shape
        phi = rsum[:, None, :].expand(R, V, T).clone()
        has_last = self.last >= 0
        if has_last.any():
            rows = torch.nonzero(has_last).squeeze(1)
            phi[rows, self.last[rows]] = self.rb[rows]                               # repeated label needs a blank in between
        first = torch.full((R, V, 1), 0.0 if self.L == 0 else NEG, dtype=torch.float64, device=self.x.device)
        phi_prev = torch.cat([first, phi[..., :-1]], -1)                             # phi_{t-1}, with phi_{-1} = log 1 for the empty prefix
        S_prev = torch.cat([torch.zeros_like(self.S[..., :1]), self.S[..., :-1]], -1)
        rn_h = self.S + torch.logcumsumexp(phi_prev - S_prev, -1)
        rn_h = torch.clamp(rn_h, min=NEG)
        rb_h = self.B[:, None, :] + _lcse_excl(rn_h - self.B[:, None, :])
        rb_h = torch.clamp(rb_h, min=NEG)
        psi = torch.logsumexp(phi_prev + self.x, -1)                                 # (R, V)
        self._rn_h, self._rb_h, self._psi = rn_h, rb_h, psi
        return psi, rsum[:, -1]

    def select(self, rows, labels):
        """advance: row i of the new state = hypothesis rows[i] extended with labels[i] (call after scores())."""
        self.rn = self._rn_h[rows, labels]
        self.rb = self._rb_h[rows, labels]
        self.psi = self._psi[rows, labels]
        self.last = labels.clone()
        self.x, self.xb, self.S, self.B = self.x[rows], self.xb[rows], self.S[rows], self.B[rows]
        self.L += 1


def ctc_full_logprob(logp, labels, blank=0):
    """reference: log P(labels | x) via torch's CTC loss, for testing. logp (T, V), labels list[int]."""
    import torch.nn.functional as F
    lp = logp.double()[:, None, :]
    return -F.ctc_loss(lp, torch.tensor([labels]), torch.tensor([lp.size(0)]), torch.tensor([len(labels)]),
                       blank=blank, reduction="sum", zero_infinity=False).item()


if __name__ == "__main__":                       # self-test: prefix states reproduce the full CTC likelihood
    torch.manual_seed(0)
    for trial in range(200):
        T, V = torch.randint(3, 25, (1,)).item(), torch.randint(3, 8, (1,)).item()
        Lmax = torch.randint(1, max(2, T // 2), (1,)).item()
        y = torch.randint(1, V, (Lmax,)).tolist()
        Tpad = T + torch.randint(0, 5, (1,)).item()
        logp = torch.log_softmax(torch.randn(1, Tpad, V), -1)
        st = ComponentCTC(logp, torch.tensor([T]), blank=0)
        for k, lab in enumerate(y):
            psi, _ = st.scores()
            st.select(torch.tensor([0]), torch.tensor([lab]))
        _, end = st.scores()
        ref = ctc_full_logprob(logp[0, :T], y)
        if ref > -1e20:
            assert abs(end.item() - ref) < 1e-6 * max(1, abs(ref)), (trial, end.item(), ref)
        # prefix probability check: psi(g) == logsumexp over all label continuations' full probs is hard; check monotonic bound
        assert st.psi.item() >= end.item() - 1e-6
    print("ctc_prefix self-test OK")
