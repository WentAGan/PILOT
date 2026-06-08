"""PILOT generative core.

This module implements the Transformer backbone and one-step flow map used by
PILOT.  The flow map learns to transport a collaborative prototype prior to the
target item embedding in a single prediction step.
"""

import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    """TF-style LayerNorm (epsilon inside the sqrt)."""

    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """Residual connection with a pre-norm."""

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.constant_(self.w_1.bias, 0)
        nn.init.xavier_normal_(self.w_2.weight)
        nn.init.constant_(self.w_2.bias, 0)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
                   for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1, 1, 1, corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _h: self.attention.forward(_h, _h, _h, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    """Stack of Transformer blocks."""

    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)]
        )

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class FlowNet(nn.Module):
    """The flow map ``Phi(x_t, t)`` that predicts the target item embedding.

    Condition injection
    -------------------
    The conditioning is made first-class and always enabled as a single
    ``cond`` mechanism:

    * ``x_t`` and the time embedding are fused into a conditioning vector;
    * the conditioning vector is prepended as a token so attention can read the
      prior directly at full strength;
    * the same vector FiLM-modulates each history position.

    If future ablations are needed, FiLM and the conditioning token should be
    removed together as ``w/o cond`` rather than judged separately.
    """

    def __init__(self, hidden_size, args):
        super(FlowNet, self).__init__()
        self.hidden_size = hidden_size
        # Sequential order is injected via learnable, right-aligned positions.
        # ML-100K histories are dense (avg ~100 items), so order carries most of
        # the signal; without this the attention is permutation-invariant and the
        # only recency cue is the prior's "last item" term.
        self.max_len = int(getattr(args, 'max_len', 50))
        self.position_embeddings = nn.Embedding(self.max_len, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, self.hidden_size),
        )
        # Fuse (flow state, time) into a single conditioning vector.
        self.cond_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        # FiLM modulation of the history positions, conditioned on (x_t, t).
        self.film = nn.Linear(self.hidden_size, self.hidden_size * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)  # -> gamma=1, beta=0 at init (identity)

        self.att = Transformer_rep(args)
        self.dropout = nn.Dropout(args.dropout)
        self.cond_norm = LayerNorm(self.hidden_size)
        self.norm_out = LayerNorm(self.hidden_size)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq):
        if t.dim() > 1:
            t = t.view(t.shape[0], -1)[:, 0]
        t = t.to(x_t.device).float()
        emb_time = self.time_embed(self.timestep_embedding(t, self.hidden_size))

        # Inject sequential order. Positions are counted from the end, so the
        # most-recent item always maps to id 0 regardless of history length; this
        # keeps the recency semantics consistent across the variable-length,
        # left-padded sequences. Padding slots receive (masked-out) high ids.
        seq_len = rep_item.shape[1]
        rev_pos = torch.arange(seq_len - 1, -1, -1, device=rep_item.device).clamp(max=self.max_len - 1)
        rep_item = rep_item + self.position_embeddings(rev_pos).unsqueeze(0)

        # Build the conditioning vector c = f(x_t, t).
        cond = self.cond_norm(self.cond_proj(torch.cat([x_t, emb_time], dim=-1)))  # [B, d]

        # Full condition injection: FiLM plus a prepended conditioning token.
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        rep_mod = rep_item * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        rep_new = torch.cat([cond.unsqueeze(1), rep_mod], dim=1)
        mask_new = torch.cat([torch.ones(mask_seq.shape[0], 1, device=mask_seq.device), mask_seq], dim=1)

        hidden = self.att(rep_new, mask_new)
        hidden = self.norm_out(self.dropout(hidden))
        out = hidden[:, -1, :]            # predicted x_1 (last real position)
        return out


class FlowMatch(nn.Module):
    """One-step flow map over the linear interpolation path.

    Training does two forwards per step:
      * the t=0 branch ``Phi(x_0, 0)`` -- this is *exactly* the one-step
        inference computation, so we optimise the deployed path directly;
      * a random path point ``Phi(x_t, t)`` with ``t`` heavy-tail sampled.
    Both are directly supervised to predict ``x_1``.
    """

    def __init__(self, args):
        super(FlowMatch, self).__init__()
        self.net = FlowNet(args.hidden_size, args)
        self.eps = float(getattr(args, 'eps', 1e-3))
        self.noise_std = float(getattr(args, 'noise_std', 0.1))
        self.s_modsamp = float(getattr(args, 's_modsamp', 1.0))

    # ---- time sampling (heavy-tailed mode sampling, as in FMRec) ------------
    def sample_t(self, batch_size, device):
        u = torch.rand(batch_size, device=device)
        correction = self.s_modsamp * (torch.cos((math.pi / 2) * u) ** 2 - 1 + u)
        t = 1 - u - correction
        return torch.clamp(t, min=self.eps, max=1.0 - self.eps)

    @staticmethod
    def interp(x0, x1, t):
        # Straight (rectified) path x_t = (1 - t) x0 + t x1.
        t = t.view(-1, 1)
        return (1.0 - t) * x0 + t * x1

    def forward(self, item_rep, x0, x1, mask_seq):
        device = x1.device
        batch_size = x1.shape[0]
        zeros_t = torch.zeros(batch_size, device=device)

        # Branch A: the exact one-step inference path Phi(x_0, 0) -> x_1.
        pred0 = self.net(item_rep, x0, zeros_t, mask_seq)

        # Branch B: a random point on the interpolation path.
        t = self.sample_t(batch_size, device)
        x_t = self.interp(x0, x1, t)
        if self.training and self.noise_std > 0:
            x_t = x_t + self.noise_std * torch.randn_like(x_t)
        pred_t = self.net(item_rep, x_t, t, mask_seq)

        flow_loss = 0.5 * (F.mse_loss(pred0, x1) + F.mse_loss(pred_t, x1))
        return pred0, flow_loss

    @torch.no_grad()
    def sample(self, item_rep, x0, mask_seq):
        # One-step inference: Phi(x_0, 0) -> predicted target embedding x_1.
        batch_size = x0.shape[0]
        t0 = torch.zeros(batch_size, device=x0.device)
        return self.net(item_rep, x0, t0, mask_seq)
