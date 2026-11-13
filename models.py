"""
Модели кандгена: общий энкодер истории + две «головы» выдачи.

  DenseRetriever — dense-ретривал (SASRec/two-tower): вектор пользователя · эмбеддинги товаров, top-K точным MIPS.
                   Обучение — sampled softmax (общие равномерные негативы на батч) по всем товарам следующей корзины.
  SIDRetriever   — генеративный ретривал в духе TIGER: по состоянию пользователя авторегрессионно генерирует
                   токены семантического ID; constrained beam search по префиксному дереву (Trie) даёт top-K.
                   Вход энкодера — сумма эмбеддингов SID-токенов товара (обобщение через общие префиксы).

Запрос = история до целевого заказа + момент запроса (время целевого заказа).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BUCKETS = 32


def time_bucket(dt_days):
    hours = torch.clamp(dt_days, min=0) * 24.0
    return torch.clamp(torch.floor(torch.log2(hours + 1.0) * 2).long(), 0, N_BUCKETS - 1)


class Encoder(nn.Module):
    """Трансформер над последовательностью товаров истории (корзины подряд); возвращает состояние на последней позиции."""

    def __init__(self, item_repr, d=64, n_layers=2, n_heads=2, max_len=100, dropout=0.1):
        super().__init__()
        self.item_repr = item_repr                     # callable: (B, L) item ids (+1, 0 = паддинг) → (B, L, d)
        self.pos = nn.Embedding(max_len, d)
        self.gap = nn.Embedding(N_BUCKETS, d)
        self.query = nn.Embedding(N_BUCKETS, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, items, times, lens, t_query):
        B, L = items.shape
        prev = torch.cat([times[:, :1], times[:, :-1]], 1)
        pos = torch.arange(L, device=items.device).unsqueeze(0).expand(B, L)
        x = self.item_repr(items) + self.pos(pos) + self.gap(time_bucket(times - prev))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=items.device), 1)
        h = self.norm(self.tf(self.drop(x), mask=causal))
        rows = torch.arange(B, device=items.device)
        last = h[rows, lens - 1]
        return last + self.query(time_bucket(t_query - times[rows, lens - 1]))


def _init(module):
    for m in module.modules():
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()


class DenseRetriever(nn.Module):
    def __init__(self, n_items, d=64, **enc_kw):
        super().__init__()
        self.n_items = n_items
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.enc = Encoder(self.item_emb, d=d, **enc_kw)
        self.out = nn.Linear(d, d)
        _init(self)

    def user(self, items, times, lens, t_query):
        return self.out(self.enc(items, times, lens, t_query))

    def loss(self, batch, n_neg):
        """batch: items, times, lens, t_query, tgt_row (P,), tgt_item (P,) — пары (запрос, товар цели)."""
        items, times, lens, tq, row, tgt = batch
        u = self.user(items, times, lens, tq)                            # (B, d)
        neg = torch.randint(1, self.n_items + 1, (n_neg,), device=u.device)
        pos_logit = (u[row] * self.item_emb(tgt)).sum(-1)                # (P,)
        neg_logit = u @ self.item_emb(neg).T                             # (B, M)
        neg_logit = neg_logit[row].masked_fill(neg.unsqueeze(0) == tgt.unsqueeze(1), -1e9)
        logits = torch.cat([pos_logit.unsqueeze(1), neg_logit], 1)
        return -F.log_softmax(logits, 1)[:, 0].mean()

    @torch.no_grad()
    def scores(self, items, times, lens, t_query):
        """(B, n_items) — скоры по всем товарам (индексы товаров с 0)."""
        return self.user(items, times, lens, t_query) @ self.item_emb.weight[1:].T

    def item_vectors(self):
        return self.item_emb.weight[1:].detach().cpu().numpy()


class SIDRetriever(nn.Module):
    """
    Декодер: последовательность [u, e_0(c_0), ..., e_{T-2}(c_{T-2})] → на позиции t предсказывает c_t.
    u — состояние пользователя из энкодера; e_l — свой словарь токенов на каждом уровне.
    """

    def __init__(self, sid, vocab, d=64, dec_layers=2, n_heads=2, dropout=0.1, use_id=False, **enc_kw):
        super().__init__()
        self.T = sid.shape[1]
        self.vocab = vocab
        self.register_buffer("sid", torch.as_tensor(np.vstack([np.zeros((1, self.T), int), sid]), dtype=torch.long))
        self.tok = nn.ModuleList([nn.Embedding(v, d) for v in vocab])
        self.id_emb = nn.Embedding(len(sid) + 1, d, padding_idx=0) if use_id else None
        self.enc = Encoder(self._item_repr, d=d, **enc_kw)
        self.u_proj = nn.Linear(d, d)
        self.lvl = nn.Embedding(self.T, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer, dec_layers, enable_nested_tensor=False)
        self.dec_norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, v) for v in vocab])
        _init(self)

    def _item_repr(self, items):
        codes = self.sid[items]                                          # (B, L, T)
        x = sum(self.tok[l](codes[..., l]) for l in range(self.T))
        x = x * (items != 0).unsqueeze(-1)
        if self.id_emb is not None:
            x = x + self.id_emb(items)
        return x

    def _decode(self, u, prefix):
        """u: (N, d); prefix: (N, l) токены уровней 0..l-1. Возвращает логиты уровня l: (N, V_l)."""
        l = prefix.shape[1]
        x = [self.u_proj(u).unsqueeze(1)] + [self.tok[j](prefix[:, j]).unsqueeze(1) for j in range(l)]
        x = torch.cat(x, 1) + self.lvl.weight[:l + 1].unsqueeze(0)
        causal = torch.triu(torch.ones(l + 1, l + 1, dtype=torch.bool, device=u.device), 1)
        h = self.dec_norm(self.dec(x, mask=causal))
        return self.heads[l](h[:, -1])

    def loss(self, batch, n_neg=None):
        items, times, lens, tq, row, tgt = batch
        u = self.enc(items, times, lens, tq)[row]                       # (P, d)
        codes = self.sid[tgt]                                            # (P, T)
        x = [self.u_proj(u).unsqueeze(1)] + [self.tok[j](codes[:, j]).unsqueeze(1) for j in range(self.T - 1)]
        x = torch.cat(x, 1) + self.lvl.weight.unsqueeze(0)
        causal = torch.triu(torch.ones(self.T, self.T, dtype=torch.bool, device=u.device), 1)
        h = self.dec_norm(self.dec(x, mask=causal))
        return sum(F.cross_entropy(self.heads[l](h[:, l]), codes[:, l]) for l in range(self.T)) / self.T

    @torch.no_grad()
    def beam_search(self, items, times, lens, t_query, trie_keys, beam):
        """
        Constrained beam search. trie_keys: список отсортированных тензоров ключей префиксов (на устройстве).
        Возвращает (full_keys (B, W), scores (B, W)) — по убыванию score; -inf у пустых лучей.
        """
        u = self.enc(items, times, lens, t_query)
        B, dev = u.shape[0], u.device
        keys = torch.zeros(B, 1, dtype=torch.long, device=dev)
        score = torch.zeros(B, 1, device=dev)
        prefix = torch.zeros(B, 1, 0, dtype=torch.long, device=dev)
        for l in range(self.T):
            W = keys.shape[1]
            V = self.vocab[l]
            logp = F.log_softmax(self._decode(u.repeat_interleave(W, 0), prefix.reshape(B * W, l)), -1)
            cand = keys.unsqueeze(-1) * V + torch.arange(V, device=dev)            # (B, W, V)
            ks = trie_keys[l]
            idx = torch.searchsorted(ks, cand.reshape(-1)).clamp_max(len(ks) - 1)
            ok = (ks[idx] == cand.reshape(-1)).reshape(B, W, V)
            total = (score.unsqueeze(-1) + logp.reshape(B, W, V)).masked_fill(~ok, -float("inf"))
            k = min(beam, W * V)
            score, flat = total.reshape(B, -1).topk(k, -1)
            parent, tok = flat // V, flat % V
            keys = cand.reshape(B, -1).gather(1, flat)
            prefix = torch.cat([prefix.gather(1, parent.unsqueeze(-1).expand(-1, -1, l)), tok.unsqueeze(-1)], -1)
        return keys, score
