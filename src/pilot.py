"""PILOT generative core."""

import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
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
        hidden = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(hidden))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError('hidden_size must be divisible by attention heads')
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
        q, k, v = [layer(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
                   for layer, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1, 1, 1, corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head)
        return self.w_layer(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(attn_heads, hidden_size, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout)
        self.input_sublayer = SublayerConnection(hidden_size, dropout)
        self.output_sublayer = SublayerConnection(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda x: self.attention(x, x, x, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class TransformerRep(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(args.hidden_size, 4, args.dropout) for _ in range(args.num_blocks)
        ])

    def forward(self, hidden, mask):
        for block in self.blocks:
            hidden = block(hidden, mask)
        return hidden


class FlowNet(nn.Module):
    def __init__(self, hidden_size, args):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = int(getattr(args, 'max_len', 50))
        self.position_embeddings = nn.Embedding(self.max_len, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, self.hidden_size),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.film = nn.Linear(self.hidden_size, self.hidden_size * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.att = TransformerRep(args)
        self.dropout = nn.Dropout(args.dropout)
        self.cond_norm = LayerNorm(self.hidden_size)
        self.norm_out = LayerNorm(self.hidden_size)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(timesteps.device)
        values = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(values), th.sin(values)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq):
        if t.dim() > 1:
            t = t.view(t.shape[0], -1)[:, 0]
        t = t.to(x_t.device).float()
        emb_time = self.time_embed(self.timestep_embedding(t, self.hidden_size))

        seq_len = rep_item.shape[1]
        rev_pos = torch.arange(seq_len - 1, -1, -1, device=rep_item.device).clamp(max=self.max_len - 1)
        rep_item = rep_item + self.position_embeddings(rev_pos).unsqueeze(0)

        cond = self.cond_norm(self.cond_proj(torch.cat([x_t, emb_time], dim=-1)))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        rep_mod = rep_item * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        rep_new = torch.cat([cond.unsqueeze(1), rep_mod], dim=1)
        mask_new = torch.cat([torch.ones(mask_seq.shape[0], 1, device=mask_seq.device), mask_seq], dim=1)
        hidden = self.att(rep_new, mask_new)
        hidden = self.norm_out(self.dropout(hidden))
        return hidden[:, -1, :]


class FlowMatch(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = FlowNet(args.hidden_size, args)
        self.eps = float(getattr(args, 'eps', 1e-3))
        self.noise_std = float(getattr(args, 'noise_std', 0.1))
        self.s_modsamp = float(getattr(args, 's_modsamp', 1.0))

    def sample_t(self, batch_size, device):
        u = torch.rand(batch_size, device=device)
        correction = self.s_modsamp * (torch.cos((math.pi / 2) * u) ** 2 - 1 + u)
        t = 1 - u - correction
        return torch.clamp(t, min=self.eps, max=1.0 - self.eps)

    @staticmethod
    def interp(x0, x1, t):
        return (1.0 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1

    def forward(self, item_rep, x0, x1, mask_seq):
        batch_size = x1.shape[0]
        device = x1.device
        zeros_t = torch.zeros(batch_size, device=device)
        pred0 = self.net(item_rep, x0, zeros_t, mask_seq)
        t = self.sample_t(batch_size, device)
        x_t = self.interp(x0, x1, t)
        if self.training and self.noise_std > 0:
            x_t = x_t + self.noise_std * torch.randn_like(x_t)
        pred_t = self.net(item_rep, x_t, t, mask_seq)
        flow_loss = 0.5 * (F.mse_loss(pred0, x1) + F.mse_loss(pred_t, x1))
        return pred0, flow_loss

    @torch.no_grad()
    def sample(self, item_rep, x0, mask_seq):
        batch_size = x0.shape[0]
        t0 = torch.zeros(batch_size, device=x0.device)
        return self.net(item_rep, x0, t0, mask_seq)
