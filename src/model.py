"""PILOT model: history encoder, prototype prior, and one-step flow map."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pilot import FlowMatch, LayerNorm


class PrototypePrior(nn.Module):
    def __init__(self, hidden_size, num_prototypes=192, heads=4, dropout=0.1, use_prototype=True):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError('hidden_size must be divisible by prior_heads')
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
        lengths = mask_seq.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (item_embeddings * mask_seq.unsqueeze(-1)).sum(dim=1) / lengths
        seq_len = item_embeddings.shape[1]
        pos = torch.arange(seq_len, device=item_embeddings.device).view(1, -1)
        last_idx = (mask_seq * pos).argmax(dim=1)
        last = item_embeddings[torch.arange(item_embeddings.shape[0], device=item_embeddings.device), last_idx]
        return 0.5 * (mean + last)

    def forward(self, item_embeddings, mask_seq):
        batch_size = item_embeddings.shape[0]
        summary = self.history_summary(item_embeddings, mask_seq)
        if not self.use_prototype:
            return self.norm(summary)

        q = self.q_proj(summary).view(batch_size, self.heads, 1, self.head_dim)
        k = self.k_proj(self.prototypes).view(1, self.heads, self.num_prototypes, self.head_dim)
        v = self.v_proj(self.prototypes).view(1, self.heads, self.num_prototypes, self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(F.softmax(scores, dim=-1))
        readout = torch.matmul(attn, v).reshape(batch_size, self.hidden_size)
        readout = self.o_proj(readout)
        return self.norm(summary + self.alpha * readout)


class PILOT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num

        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.layer_norm = LayerNorm(self.emb_dim, eps=1e-12)

        self.prior = PrototypePrior(
            self.emb_dim,
            num_prototypes=int(getattr(args, 'num_prototypes', 192)),
            heads=int(getattr(args, 'prior_heads', 4)),
            dropout=args.dropout,
            use_prototype=bool(int(getattr(args, 'use_prototype', 1))),
        )
        self.flow = FlowMatch(args)
        self.prior_proj = nn.Linear(self.emb_dim, self.emb_dim)

        self.flow_weight = float(getattr(args, 'flow_weight', 1.0))
        self.ce_weight = float(getattr(args, 'ce_weight', 0.5))
        self.prior_ce_weight = float(getattr(args, 'prior_ce_weight', 0.3))

        self.apply(self._init_weights)
        nn.init.zeros_(self.flow.net.film.weight)
        nn.init.zeros_(self.flow.net.film.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def encode_history(self, sequence):
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.layer_norm(item_embeddings)
        mask_seq = (sequence > 0).float()
        return item_embeddings, mask_seq

    def score(self, rep):
        return torch.matmul(rep, self.item_embeddings.weight.t())

    def ce_loss(self, rep, labels):
        scores = self.score(rep)
        return F.cross_entropy(scores, labels.squeeze(-1))

    def prior_ce_loss(self, x0, labels):
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
        total = self.flow_weight * flow_loss + self.ce_weight * ce + self.prior_ce_weight * prior_ce
        parts = {'flow': flow_loss.detach(), 'ce': ce.detach(), 'prior_ce': prior_ce.detach()}
        return pred0, total, parts


def create_model(args):
    return PILOT(args)
