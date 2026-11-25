import torch, torch.nn as nn
D, L, CTX = 512, 6, 64
class LM(nn.Module):
    def __init__(s, V, ctx=CTX):
        super().__init__()
        s.emb = nn.Embedding(V, D); s.pos = nn.Embedding(ctx, D)
        layer = nn.TransformerEncoderLayer(D, 8, 2048, 0.1, batch_first=True, norm_first=True)
        s.tr = nn.TransformerEncoder(layer, L, norm=nn.LayerNorm(D), enable_nested_tensor=False)
        s.out = nn.Linear(D, V, bias=False); s.out.weight = s.emb.weight
        nn.init.normal_(s.emb.weight, std=D ** -0.5); nn.init.normal_(s.pos.weight, std=0.02)
    def forward(s, x):
        T = x.size(1)
        h = s.emb(x) + s.pos(torch.arange(T, device=x.device))
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        return s.out(s.tr(h, mask=m, is_causal=True))
