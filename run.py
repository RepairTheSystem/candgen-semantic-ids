"""
Эксперимент: кандген для маркетплейса с повторами — Instacart.

Источники кандидатов (top-K, K до --kmax):
  pop          — популярность в train;
  hist         — товары из истории пользователя по частоте с затуханием по заказам (только повторы);
  hist+pop     — hist, остаток — популярное (классический бейзлайн);
  dense        — dense-ретривал (SASRec/two-tower, sampled softmax), точный MIPS по каталогу;
  dense-exp    — то же без товаров истории (источник «нового»);
  sid-<src>    — генеративный ретривал по семантическим ID (src: text — из текста товара, cf — из эмбеддингов dense);
  sid-<src>-exp, hybrid-<src> (SID-кандидаты → доранжирование dense-скором, как LIGER);
  union[...]   — K слотов: q из hist + (K − q) из источника «нового»; q — подобранная на val константа (fixed),
                 выученный переключатель на запрос (switch: регрессор доли повторов → q = α·K·p̂) или оракул.

Гипотезы: H1 — SID лучше на новом/хвосте, ID-модели и история — на повторах;
          H2 — переключатель между источниками на запрос лучше фиксированной квоты.

  python run.py --fake                               # дымовой прогон на фейковых данных формата Instacart
  python run.py --data-root <папка с csv> --n-users 30000
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from instacart import find_root, item_counts, load_instacart, make_fake_instacart, make_protocol
from models import DenseRetriever, SIDRetriever
from semantic_ids import Trie, build_sids, normed, text_vectors


# ----------------------------------------------------------------------------- батчи


def user_arrays(events):
    return {u: (g.item.to_numpy() + 1, g.t.to_numpy(), g.order.to_numpy()) for u, g in events.groupby("user")}


def train_samples(train, max_len, per_user):
    """Обучающие запросы: (история до заказа j, время заказа j, товары заказа j) для последних per_user заказов."""
    out = []
    for u, (it, ts, od) in user_arrays(train).items():
        bounds = np.flatnonzero(np.diff(od)) + 1                       # начала заказов 1..n-1
        for s in bounds[-per_user:]:
            e = np.searchsorted(od, od[s], side="right")
            h0 = max(0, s - max_len)
            out.append((it[h0:s], ts[h0:s], ts[s], np.unique(it[s:e])))
    return out


def collate(samples, max_len, device):
    B = len(samples)
    items = np.zeros((B, max_len), np.int64)
    times = np.zeros((B, max_len), np.float32)
    lens = np.zeros(B, np.int64)
    tq = np.zeros(B, np.float32)
    rows, tgts = [], []
    for b, (it, ts, t, tg) in enumerate(samples):
        n = len(it)
        items[b, :n], times[b, :n], times[b, n:], lens[b], tq[b] = it, ts, ts[-1], n, t
        if tg is not None:
            rows.append(np.full(len(tg), b))
            tgts.append(tg)
    T = lambda a: torch.from_numpy(a).to(device)
    batch = [T(items), T(times), T(lens), T(tq)]
    if rows:
        batch += [T(np.concatenate(rows)), T(np.concatenate(tgts).astype(np.int64))]
    return batch


def query_samples(split, users, max_len):
    return [((split.hist[u][0][-max_len:] + 1), split.hist[u][1][-max_len:], split.t_query[u], None) for u in users]


# ----------------------------------------------------------------------------- метрики


def recall_table(ranked, users, split, head, ks):
    """ranked: (U, ≥K) id товаров (−1 — пусто). Recall по пользователям + микро по repeat / explore / explore-хвост."""
    out = {}
    tg = [split.targets[u] for u in users]
    rep = [np.isin(t, split.hist[u][0]) for t, u in zip(tg, users)]
    tail = [~head[t] for t in tg]
    for k in ks:
        hit = [np.isin(t, ranked[r, :k]) for r, t in enumerate(tg)]
        n = lambda m: max(sum(x.sum() for x in m), 1)
        out[f"R@{k}"] = float(np.mean([h.mean() for h in hit]))
        out[f"R@{k} repeat"] = float(sum((h & r).sum() for h, r in zip(hit, rep)) / n(rep))
        out[f"R@{k} explore"] = float(sum((h & ~r).sum() for h, r in zip(hit, rep)) / n([~r for r in rep]))
        out[f"R@{k} explore-tail"] = float(sum((h & ~r & tl).sum() for h, r, tl in zip(hit, rep, tail))
                                           / n([~r & tl for r, tl in zip(rep, tail)]))
    return out


def recall_at(ranked, users, split, k):
    return float(np.mean([np.isin(split.targets[u], ranked[r, :k]).mean() for r, u in enumerate(users)]))


# ----------------------------------------------------------------------------- источники


def pad(lists, k):
    out = np.full((len(lists), k), -1, np.int64)
    for r, l in enumerate(lists):
        l = l[:k]
        out[r, :len(l)] = l
    return out


def hist_rank(split, users, decay=0.9):
    """История пользователя: Σ decay^(заказов назад) по покупкам товара."""
    res = []
    for u in users:
        it, _, od = split.hist[u]
        w = decay ** (od.max() - od)
        s = pd.Series(w).groupby(it).sum().sort_values(ascending=False)
        res.append(s.index.to_numpy())
    return res


def fill(primary, filler, k):
    """primary (списки) + остаток из filler (общий ранжированный массив), без повторов."""
    out = []
    for p in primary:
        seen = set(p.tolist())
        rest = [i for i in filler[:k + len(p)] if i not in seen][:max(0, k - len(p))]
        out.append(np.concatenate([p[:k], np.array(rest, np.int64)]))
    return pad(out, k)


def topk_scores(scores, k, exclude=None):
    s = scores.copy()
    if exclude is not None:
        for r, ex in enumerate(exclude):
            s[r, ex] = -np.inf
    idx = np.argpartition(-s, k, 1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, idx, 1), 1)
    idx = np.take_along_axis(idx, order, 1)
    return np.where(np.isfinite(np.take_along_axis(s, idx, 1)), idx, -1)


@torch.no_grad()
def dense_rank(model, split, users, args, k, exclude_hist=False, cands=None):
    """
    Батчами, без матрицы (U, N) в памяти: top-k dense-скоров (exclude_hist — без товаров истории)
    или, если заданы cands (U, M), — доранжирование этих кандидатов dense-скором.
    """
    model.eval()
    outs = []
    for s in range(0, len(users), args.eval_batch):
        us = users[s:s + args.eval_batch]
        b = collate(query_samples(split, us, args.max_len), args.max_len, args.device)
        sc = model.scores(*b).float()
        if cands is not None:
            c = torch.as_tensor(cands[s:s + len(us)], device=sc.device)
            cs = sc.gather(1, c.clamp_min(0)).masked_fill(c < 0, -float("inf"))
            order = cs.argsort(1, descending=True)
            outs.append(c.gather(1, order).cpu().numpy())
            continue
        if exclude_hist:
            for r, u in enumerate(us):
                sc[r, torch.as_tensor(np.unique(split.hist[u][0]), device=sc.device)] = -float("inf")
        val, idx = sc.topk(k, 1)
        outs.append(torch.where(torch.isfinite(val), idx, -1).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def sid_beam(model, trie, trie_keys, split, users, args, beam):
    """(U, beam) товаров по убыванию лог-вероятности SID."""
    model.eval()
    outs = []
    for s in range(0, len(users), args.beam_batch):
        b = collate(query_samples(split, users[s:s + args.beam_batch], args.max_len), args.max_len, args.device)
        keys, score = model.beam_search(*b, trie_keys, beam)
        items = trie.items(keys.cpu().numpy())
        items[~np.isfinite(score.cpu().numpy())] = -1
        outs.append(pad(list(items), beam))
    return np.concatenate(outs)


def drop_hist(ranked, split, users, k):
    return pad([r[(r >= 0) & ~np.isin(r, split.hist[u][0])] for r, u in zip(ranked, users)], k)


def union(hist_lists, exp_ranked, q, k):
    q = np.broadcast_to(np.asarray(q), (len(hist_lists),))
    out = []
    for r, h in enumerate(hist_lists):
        m = int(min(q[r], len(h), k))
        e = exp_ranked[r][exp_ranked[r] >= 0][:k - m]
        out.append(np.concatenate([h[:m], e]))
    return pad(out, k)


# ----------------------------------------------------------------------------- переключатель


def user_features(split, users):
    rows = []
    for u in users:
        it, ts, od = split.hist[u]
        bounds = np.r_[0, np.flatnonzero(np.diff(od)) + 1, len(od)]
        seen, rep_share = set(), []
        for s, e in zip(bounds[:-1], bounds[1:]):
            b = set(it[s:e].tolist())
            if seen:
                rep_share.append(len(b & seen) / len(b))
            seen |= b
        gaps = np.diff(ts[bounds[:-1]]) if len(bounds) > 2 else np.array([0.0])
        rows.append([len(bounds) - 1, len(seen), len(it) / (len(bounds) - 1),
                     np.mean(rep_share) if rep_share else 0.0, rep_share[-1] if rep_share else 0.0,
                     split.t_query[u] - ts[-1], gaps.mean(), gaps.std(), bounds[-1] - bounds[-2]])
    return np.array(rows, float)


def repeat_frac(split, users):
    return np.array([np.isin(split.targets[u], split.hist[u][0]).mean() for u in users])


def fit_switch(train_events, users_seed):
    """Регрессор доли повторов в следующей корзине: обучается на последнем заказе train (история — до него)."""
    _, _, pseudo = make_protocol(train_events)          # цель = последний train-заказ, история — до него
    us = pseudo.users
    if len(us) > 20000:
        us = np.random.default_rng(users_seed).choice(us, 20000, replace=False)
    reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=users_seed)
    reg.fit(user_features(pseudo, us), repeat_frac(pseudo, us))
    return reg


# ----------------------------------------------------------------------------- обучение


def train_model(model, samples, val, val_users, args, eval_fn, name):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(args.max_epochs):
        model.train()
        random.shuffle(samples)
        tot, n, t0 = 0.0, 0, time.time()
        for s in range(0, len(samples), args.batch):
            loss = model.loss(collate(samples[s:s + args.batch], args.max_len, args.device), args.n_neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item(), n + 1
        rec = eval_fn(model)
        print(f"  [{name}] epoch {ep + 1} loss {tot / n:.4f} val R@50 {rec:.4f} ({time.time() - t0:.0f} с)", flush=True)
        if rec > best + 1e-4:
            best, best_state, best_ep, bad = rec, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best_ep, best


# ----------------------------------------------------------------------------- латентность


def sync(args):
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()


def online_ms(fn, users, n=100, warmup=10):
    ms = []
    for i, u in enumerate(users[:n + warmup]):
        t = time.perf_counter()
        fn(np.array([u]))
        if i >= warmup:
            ms.append(1000 * (time.perf_counter() - t))
    return {f"online p{q} ms": float(np.percentile(ms, q)) for q in (50, 95, 99)}


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--fake", action="store_true", help="сгенерировать фейковые данные формата Instacart")
    ap.add_argument("--fake-users", type=int, default=3000)
    ap.add_argument("--n-users", type=int, default=30000)
    ap.add_argument("--min-item-count", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--per-user", type=int, default=5, help="обучающих заказов на пользователя")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--beam-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-neg", type=int, default=2048)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--sid-codes", type=int, nargs="+", default=[128, 128, 128])
    ap.add_argument("--sid-sources", nargs="+", default=["text", "cf"], choices=["text", "cf", "text+cf"])
    ap.add_argument("--kmax", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100, 200])
    ap.add_argument("--val-users", type=int, default=3000, help="пользователей val для ранней остановки")
    ap.add_argument("--head-share", type=float, default=0.2, help="доля самых популярных товаров = «голова»")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    K, ks = args.kmax, [k for k in args.ks if k <= args.kmax]
    log = {"args": vars(args)}

    # --- данные
    t0 = time.time()
    if args.fake:
        root = make_fake_instacart(os.path.join(args.out, "fake_data"), n_users=args.fake_users, seed=args.seed)
    else:
        root = find_root(args.data_root)
    events, items = load_instacart(root, n_users=args.n_users, min_item_count=args.min_item_count, seed=args.seed)
    train, val, test = make_protocol(events)
    N = len(items)
    cnt = item_counts(train, N)
    head = np.zeros(N, bool)
    head[np.argsort(-cnt)[:max(1, int(args.head_share * N))]] = True
    vu_all, tu = val.users, test.users
    vu = vu_all if len(vu_all) <= args.val_users else np.random.default_rng(args.seed).choice(vu_all, args.val_users, replace=False)
    rf = repeat_frac(test, tu)
    log["data"] = {"root": root, "events": len(events), "users": int(events.user.nunique()), "items": N,
                   "train events": len(train), "test users": len(tu),
                   "test basket size": float(np.mean([len(test.targets[u]) for u in tu])),
                   "repeat share (test, по товарам)": float(np.concatenate(
                       [np.isin(test.targets[u], test.hist[u][0]) for u in tu]).mean()),
                   "distinct history items (медиана)": float(np.median([len(np.unique(test.hist[u][0])) for u in tu])),
                   "load sec": round(time.time() - t0)}
    print(json.dumps(log["data"], ensure_ascii=False, indent=1), flush=True)
    samples = train_samples(train, args.max_len, args.per_user)
    print(f"обучающих запросов: {len(samples)}", flush=True)
    enc_kw = dict(n_layers=args.layers, max_len=args.max_len)

    R = {}                                                     # имя → ранжированный массив (U_test, K)
    lat = []

    # --- простые источники
    pop = np.argsort(-cnt)
    hist_t = hist_rank(test, tu)
    R["pop"] = np.tile(pop[:K], (len(tu), 1))
    R["hist"] = pad(hist_t, K)
    R["hist+pop"] = fill(hist_t, pop, K)

    # --- dense
    t0 = time.time()
    dense = DenseRetriever(N, d=args.dim, **enc_kw).to(args.device)
    dense, ep, _ = train_model(dense, list(samples), val, vu, args,
                               lambda m: recall_at(dense_rank(m, val, vu, args, 50), vu, val, 50), "dense")
    log["dense"] = {"best epoch": ep, "train sec": round(time.time() - t0)}
    R["dense"] = dense_rank(dense, test, tu, args, K)
    R["dense-exp"] = dense_rank(dense, test, tu, args, K, exclude_hist=True)

    # --- семантические ID и генеративный ретривал
    vecs = {}
    for src in args.sid_sources:
        if src == "text":
            vecs[src] = text_vectors(items, dim=64, seed=args.seed)
        elif src == "cf":
            vecs[src] = normed(dense.item_vectors())
        else:
            vecs[src] = normed(np.hstack([text_vectors(items, 64, args.seed), normed(dense.item_vectors())]))
    log["sid"] = {}
    sid_models = {}
    for src, Z in vecs.items():
        t0 = time.time()
        S = build_sids(Z, tuple(args.sid_codes), seed=args.seed)
        trie = Trie(S["sid"], S["vocab"])
        tk = [torch.as_tensor(k, device=args.device) for k in trie.keys]
        m = SIDRetriever(S["sid"], S["vocab"], d=args.dim, **enc_kw).to(args.device)
        m, ep, _ = train_model(m, list(samples), val, vu, args,
                               lambda mm: recall_at(sid_beam(mm, trie, tk, val, vu, args, 50), vu, val, 50), f"sid-{src}")
        log["sid"][src] = {**S["stats"], "best epoch": ep, "train sec": round(time.time() - t0)}
        sid_models[src] = (m, trie, tk)
        beam = sid_beam(m, trie, tk, test, tu, args, 2 * K)
        R[f"sid-{src}"] = beam[:, :K]
        R[f"sid-{src}-exp"] = drop_hist(beam, test, tu, K)
        R[f"hybrid-{src}"] = dense_rank(dense, test, tu, args, K, cands=beam)[:, :K]   # LIGER-подобный гибрид
        print(f"  sid-{src}: {json.dumps(S['stats'], ensure_ascii=False)}", flush=True)

    # --- объединение источников: фиксированная квота vs переключатель vs оракул
    reg = fit_switch(train, args.seed)
    p_val, p_test = reg.predict(user_features(val, vu_all)).clip(0, 1), reg.predict(user_features(test, tu)).clip(0, 1)
    rf_val = repeat_frac(val, vu_all)
    log["switch"] = {"corr(p̂, доля повторов) val": float(np.corrcoef(p_val, rf_val)[0, 1]),
                     "corr test": float(np.corrcoef(p_test, rf)[0, 1])}
    hist_v = hist_rank(val, vu_all)
    exp_sources = {"dense-exp": None}
    exp_sources.update({f"sid-{s}-exp": s for s in sid_models})
    union_rows = []
    for name, src in exp_sources.items():
        if src is None:
            ev = dense_rank(dense, val, vu_all, args, K, exclude_hist=True)
        else:
            m, trie, tk = sid_models[src]
            ev = drop_hist(sid_beam(m, trie, tk, val, vu_all, args, 2 * K), val, vu_all, K)
        et = R[name]
        for k in ks:
            grid_q = sorted({int(round(k * f)) for f in np.linspace(0, 1, 11)})
            q_best = max(grid_q, key=lambda q: recall_at(union(hist_v, ev, q, k), vu_all, val, k))
            grid_a = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
            qa = lambda p, a: np.round(np.minimum(1.0, a * p) * k)
            a_best = max(grid_a, key=lambda a: recall_at(union(hist_v, ev, qa(p_val, a), k), vu_all, val, k))
            for kind, q in (("fixed", q_best), ("switch", qa(p_test, a_best)), ("oracle", qa(rf, 1.0)),
                            ("hist-first", k)):
                r = recall_table(union(hist_t, et, q, k), tu, test, head, [k])
                union_rows.append({"exp source": name, "K": k, "kind": kind,
                                   "param": q_best if kind == "fixed" else (a_best if kind == "switch" else ""),
                                   **{c.replace(f"@{k}", "@K"): v for c, v in r.items()}})
    un = pd.DataFrame(union_rows)

    # --- метрики источников
    rows = [{"source": n, **recall_table(r, tu, test, head, ks)} for n, r in R.items()]
    for n, r in R.items():
        rows[[x["source"] for x in rows].index(n)]["coverage@100"] = len(np.unique(r[:, :100][r[:, :100] >= 0])) / N
    res = pd.DataFrame(rows).set_index("source")

    # --- латентность (batch=1, полный путь до top-K)
    one = argparse.Namespace(**{**vars(args), "eval_batch": 1, "beam_batch": 1})

    def timed_online(fn):
        def wrap(u):
            fn(u)
            sync(args)
        return online_ms(wrap, tu)

    lat.append({"source": "hist+pop", **timed_online(lambda u: fill(hist_rank(test, u), pop, K))})
    lat.append({"source": "dense", **timed_online(lambda u: dense_rank(dense, test, u, one, K))})
    for src, (m, trie, tk) in sid_models.items():
        lat.append({"source": f"sid-{src} (beam {K})",
                    **timed_online(lambda u, m=m, trie=trie, tk=tk: sid_beam(m, trie, tk, test, u, one, K))})
        lat.append({"source": f"hybrid-{src} (beam {2 * K} + dense)",
                    **timed_online(lambda u, m=m, trie=trie, tk=tk: dense_rank(
                        dense, test, u, one, K, cands=sid_beam(m, trie, tk, test, u, one, 2 * K)))})
    lat = pd.DataFrame(lat).set_index("source")
    for name, m in [("dense", dense)] + [(f"sid-{s}", v[0]) for s, v in sid_models.items()]:
        log.setdefault("params", {})[name] = sum(p.numel() for p in m.parameters())

    # --- отчёт
    res.to_csv(os.path.join(args.out, "sources.csv"))
    un.to_csv(os.path.join(args.out, "union.csv"), index=False)
    lat.to_csv(os.path.join(args.out, "latency.csv"))
    with open(os.path.join(args.out, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1, default=str)
    md = lambda df, **kw: df.to_markdown(**kw) if _has_tabulate() else "```\n" + df.to_string() + "\n```"
    k0 = 100 if 100 in ks else ks[-1]
    lines = [f"# Кандген на Instacart ({'фейк' if args.fake else 'реальные данные'}), устройство {args.device}\n",
             "## Данные\n", "```json\n" + json.dumps(log["data"], ensure_ascii=False, indent=1) + "\n```\n",
             "## Семантические ID\n", "```json\n" + json.dumps(log["sid"], ensure_ascii=False, indent=1) + "\n```\n",
             f"## Источники: Recall@K (тест)\n", md(res[[f"R@{k}" for k in ks]], floatfmt=".4f"),
             f"\n\n## Разбивка при K={k0}: повторы / новое / новое из хвоста\n",
             md(res[[f"R@{k0}", f"R@{k0} repeat", f"R@{k0} explore", f"R@{k0} explore-tail", "coverage@100"]],
                floatfmt=".4f"),
             "\n\n## Объединение источников: hist + источник нового\n",
             "`fixed` — квота q подобрана на val; `switch` — q = α·K·p̂ (p̂ — регрессор доли повторов, α на val); "
             "`oracle` — q = K·(истинная доля повторов); `hist-first` — вся история, затем новое.\n",
             "```json\n" + json.dumps(log["switch"], ensure_ascii=False) + "\n```\n",
             md(un.set_index(["exp source", "K", "kind"]), floatfmt=".4f"),
             f"\n\n## Латентность (batch=1, {args.device})\n", md(lat, floatfmt=".2f"),
             "\n\n## Обучение\n", "```json\n" + json.dumps({"dense": log["dense"], "params": log["params"]},
                                                           ensure_ascii=False, indent=1) + "\n```\n"]
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
