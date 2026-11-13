"""
Семантические ID (SID): товар → короткая последовательность токенов (c1, c2, ..., cL, d).

  1. Вектор товара:
       text — TF-IDF по «название | ряд | отдел» → TruncatedSVD → L2-норма (работает без интернета и внешних моделей);
       cf   — эмбеддинги товаров из обученной dense-модели (коллаборативные SID);
       text+cf — конкатенация нормированных векторов.
  2. Residual k-means (RQ-KMeans, как в OneRec): уровень l кластеризует остаток после уровней < l.
     Близкие товары делят префикс — на этом держится обобщение генеративного ретривала.
  3. Последний токен d — номер товара внутри группы с одинаковым (c1..cL), чтобы SID был уникален (как в TIGER).
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def text_vectors(items, dim=64, seed=0):
    text = (items["name"].astype(str) + " | " + items["aisle"].astype(str) + " | "
            + items["department"].astype(str)).str.lower()
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(text)
    dim = min(dim, X.shape[1] - 1)
    Z = TruncatedSVD(dim, random_state=seed).fit_transform(X)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def normed(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def residual_kmeans(Z, codes=(128, 128, 128), seed=0):
    """Z: (N, d). Возвращает (N, L) коды и остаточную ошибку после каждого уровня."""
    R = Z.astype(np.float64).copy()
    out, err = [], []
    for l, k in enumerate(codes):
        k = min(k, len(R))
        km = MiniBatchKMeans(k, random_state=seed + l, batch_size=4096, n_init=3).fit(R)
        c = km.predict(R)
        R = R - km.cluster_centers_[c]
        out.append(c)
        err.append(float((R ** 2).sum(1).mean()))
    return np.stack(out, 1), err


def add_dedup(codes):
    """Добавляет токен-разделитель коллизий: номер товара внутри группы одинаковых кодов."""
    _, inv = np.unique(codes, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    d = np.empty(len(inv), dtype=np.int64)
    first = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]
    for s, e in zip(first, np.r_[first[1:], len(inv)]):
        d[order[s:e]] = np.arange(e - s)
    return np.concatenate([codes, d[:, None]], 1)


def build_sids(Z, codes=(128, 128, 128), seed=0):
    """Возвращает dict: sid (N, L+1), vocab (размер словаря на уровне), stats."""
    c, err = residual_kmeans(Z, codes, seed)
    sid = add_dedup(c)
    vocab = [int(sid[:, l].max()) + 1 for l in range(sid.shape[1])]
    groups = np.unique(c, axis=0, return_counts=True)[1]
    stats = {"levels": list(codes), "vocab": vocab, "unique codes (без d)": int(len(groups)),
             "доля товаров с коллизией": float((groups[groups > 1]).sum() / len(sid)),
             "макс. коллизия": int(groups.max()), "residual err по уровням": [round(e, 4) for e in err],
             "использовано кодов уровня 1": int(len(np.unique(c[:, 0])))}
    return {"sid": sid, "vocab": vocab, "stats": stats}


class Trie:
    """
    Допустимые продолжения префиксов для constrained beam search.
    Префикс уровня l кодируется числом key_l = key_{l-1} · V_l + c_l; допустимость проверяется
    бинарным поиском по отсортированным ключам реальных товаров.
    """

    def __init__(self, sid, vocab):
        self.vocab = vocab
        self.L = sid.shape[1]
        self.keys = []                                  # отсортированные ключи префиксов длины l+1
        key = np.zeros(len(sid), dtype=np.int64)
        for l in range(self.L):
            key = key * vocab[l] + sid[:, l]
            self.keys.append(np.unique(key))
        order = np.argsort(key)
        self.full_keys, self.item_of_key = key[order], order    # полный ключ → товар

    def key_extend(self, prefix_key, l, tokens):
        return prefix_key * self.vocab[l] + tokens

    def valid(self, keys, l):
        """keys: np.int64 любой формы — ключи префиксов длины l+1. Возвращает bool той же формы."""
        ks = self.keys[l]
        idx = np.clip(np.searchsorted(ks, keys), 0, len(ks) - 1)
        return ks[idx] == keys

    def items(self, full_keys):
        idx = np.clip(np.searchsorted(self.full_keys, full_keys), 0, len(self.full_keys) - 1)
        ok = self.full_keys[idx] == full_keys
        return np.where(ok, self.item_of_key[idx], -1)
