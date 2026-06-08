"""PILOT model: history encoder + Collaborative Prototype Prior + flow map.

The prior starts from a personalised history summary and adds a learnable
collaborative prototype read-out.  The flow map then transports this residual
prototype prior to the target item embedding in a single step.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pilot import FlowMatch, LayerNorm


class PrototypePrior(nn.Module):
    """Residual collaborative-prototype prior.

    x_0 = LayerNorm(history_summary + alpha * prototype_readout)

    * ``history_summary`` keeps each user's own signal and is well defined even
      for a one-item history;
    * the multi-head prototype read-out adds cross-user collaborative structure;
    * ``alpha`` is a learnable per-dimension gate, initialised small, so the
      model controls how much collaborative prior to mix in.

    Ablation: ``use_prototype=False`` removes the prototype read-out entirely,
    so ``x_0`` falls back to ``LayerNorm(history_summary)``.
    """

    def __init__(self, hidden_size, num_prototypes=128, heads=4, dropout=0.1, use_prototype=True):
        super(PrototypePrior, self).__init__()
        assert hidden_size % heads == 0, 'hidden_size must be divisible by prior_heads'
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.use_prototype = use_prototype
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_size))
        nn.init.normal_(self.prototypes, std=0.02)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.alpha = nn.Parameter(torch.full((hidden_size,), 0.1))
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def history_summary(self, item_embeddings, mask_seq):
        """Length-robust history summary: mean over valid positions plus the
        last valid item (recency).  Works for any history length >= 1."""
        lengths = mask_seq.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (item_embeddings * mask_seq.unsqueeze(-1)).sum(dim=1) / lengths
        seq_len = item_embeddings.shape[1]
        pos = torch.arange(seq_len, device=item_embeddings.device).view(1, -1)
        last_idx = (mask_seq * pos).argmax(dim=1)
        last = item_embeddings[torch.arange(item_embeddings.shape[0], device=item_embeddings.device), last_idx]
        return 0.5 * (mean + last)

    def forward(self, item_embeddings, mask_seq):
        b = item_embeddings.shape[0]
        summary = self.history_summary(item_embeddings, mask_seq)             # [B, d]

        if not self.use_prototype:                                            # ablation: no prototype read-out
            return self.norm(summary)

        # Multi-head attention read-out over the shared prototype bank.
        q = self.q_proj(summary).view(b, self.heads, 1, self.head_dim)
        k = self.k_proj(self.prototypes).view(1, self.heads, self.num_prototypes, self.head_dim)
        v = self.v_proj(self.prototypes).view(1, self.heads, self.num_prototypes, self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(F.softmax(scores, dim=-1))                        # [B, H, 1, K]
        readout = torch.matmul(attn, v).reshape(b, self.hidden_size)
        readout = self.o_proj(readout)

        prior = summary + self.alpha * readout
        return self.norm(prior)


class PILOT(nn.Module):
    def __init__(self, args):
        super(PILOT, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num

        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.LayerNorm = LayerNorm(self.emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)

        self.prior = PrototypePrior(self.emb_dim,
                                    num_prototypes=int(getattr(args, 'num_prototypes', 128)),
                                    heads=int(getattr(args, 'prior_heads', 4)),
                                    dropout=args.dropout,
                                    use_prototype=bool(int(getattr(args, 'use_prototype', 1))))
        self.flow = FlowMatch(args)

        # Source-side anchor: predict the next item directly from the prior path.
        self.prior_proj = nn.Linear(self.emb_dim, self.emb_dim)

        self.flow_weight = float(getattr(args, 'flow_weight', 1.0))
        self.prior_ce_weight = float(getattr(args, 'prior_ce_weight', 0.3))
        self.ce_weight = float(getattr(args, 'ce_weight', 0.5))

        self.apply(self._init_weights)

        # ``apply`` re-initialises every nn.Linear, which silently overwrites the
        # zero-init of the FiLM projection in FlowNet.  Restore it so the
        # modulation truly starts at identity (gamma=1, beta=0): the history then
        # enters the flow net unscaled at step 0 and the network learns to
        # modulate gradually.  This removes a per-seed source of init noise that
        # showed up as the high ML-100K variance.
        nn.init.zeros_(self.flow.net.film.weight)
        nn.init.zeros_(self.flow.net.film.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # -- encoding ---------------------------------------------------------------
    def encode_history(self, sequence):
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.LayerNorm(item_embeddings)
        mask_seq = (sequence > 0).float()
        return item_embeddings, mask_seq

    # -- scoring (evaluation path: pure inner product, protocol-identical) ------
    def score(self, rep):
        return torch.matmul(rep, self.item_embeddings.weight.t())

    # -- losses -----------------------------------------------------------------
    def ce_loss(self, rep, labels):
        scores = self.score(rep)
        return F.cross_entropy(scores, labels.squeeze(-1))

    def prior_ce_loss(self, x0, labels):
        """Next-item CE computed directly from the collaborative prior x_0."""
        rep = self.prior_proj(x0)
        return self.ce_loss(rep, labels)

    def forward(self, sequence, tag, train_flag=True):
        item_embeddings, mask_seq = self.encode_history(sequence)
        x0 = self.prior(item_embeddings, mask_seq)

        if not train_flag:
            return self.flow.sample(item_embeddings, x0, mask_seq)

        x1 = self.item_embeddings(tag.squeeze(-1))
        pred0, flow_loss = self.flow(item_embeddings, x0, x1, mask_seq)

        ce = self.ce_loss(pred0, tag)
        prior_ce = self.prior_ce_loss(x0, tag)

        total = (self.flow_weight * flow_loss
                 + self.ce_weight * ce
                 + self.prior_ce_weight * prior_ce)
        parts = {'flow': flow_loss.detach(),
                 'ce': ce.detach(), 'prior_ce': prior_ce.detach()}
        return pred0, total, parts


def create_model(args):
    return PILOT(args)
