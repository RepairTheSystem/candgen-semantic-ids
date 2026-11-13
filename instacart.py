"""
Instacart: загрузка, фейковые данные того же формата (для локальных проверок) и протокол оценки.

Файлы Instacart Market Basket Analysis (Kaggle), можно и в .csv.zip:
  orders.csv                 order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order
  order_products__prior.csv  order_id, product_id, add_to_cart_order, reordered
  order_products__train.csv  то же
  products.csv               product_id, product_name, aisle_id, department_id
  aisles.csv, departments.csv

Особенности данных:
  * календарных дат нет — время восстанавливаем как сумму days_since_prior_order по заказам пользователя;
    значения обрезаны сверху на 30 («30» = «30 и больше»);
  * eval_set == "test" — заказы без состава (это тест соревнования), их выбрасываем.

Протокол (стандартный для next-basket на Instacart — глобального времени нет):
  test-цель = последний заказ пользователя, val-цель = предпоследний;
  модель учится на заказах до val-цели; история для теста включает val-заказ (модель не переобучается).
  Цель — множество различных товаров заказа; repeat — товар уже был в истории пользователя.
"""
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

FILES = ["orders", "order_products__prior", "order_products__train", "products", "aisles", "departments"]


def find_root(path=None):
    """Папка, где лежит orders.csv(.zip): явный путь или поиск в /kaggle/input."""
    cands = [path] if path else []
    cands += sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/orders.csv*", recursive=True)})
    for c in cands:
        if c and (os.path.exists(os.path.join(c, "orders.csv")) or os.path.exists(os.path.join(c, "orders.csv.zip"))):
            return c
    raise FileNotFoundError("не нашёл orders.csv: укажи --data-root или подключи датасет Instacart в Kaggle")


def _read(root, name, **kw):
    for ext in (".csv", ".csv.zip"):
        p = os.path.join(root, name + ext)
        if os.path.exists(p):
            return pd.read_csv(p, **kw)
    raise FileNotFoundError(os.path.join(root, name + ".csv"))


def load_instacart(root, n_users=None, min_item_count=10, min_orders=4, seed=0):
    """
    Возвращает:
      events: DataFrame[user, item, order, t, pos] — user/item/order с 0, t — дни от первого заказа, pos — порядок в корзине;
      items:  DataFrame[item, product_id, name, aisle, department].
    n_users — случайная подвыборка пользователей (для скорости); min_item_count — порог частоты товара.
    """
    orders = _read(root, "orders", usecols=["order_id", "user_id", "eval_set", "order_number",
                                            "days_since_prior_order"])
    orders = orders[orders.eval_set != "test"]
    users = orders.user_id.unique()
    if n_users and n_users < len(users):
        users = np.random.default_rng(seed).choice(users, n_users, replace=False)
        orders = orders[orders.user_id.isin(users)]
    orders = orders.sort_values(["user_id", "order_number"])
    orders["t"] = orders.days_since_prior_order.fillna(0).groupby(orders.user_id).cumsum()

    cols = ["order_id", "product_id", "add_to_cart_order"]
    op = pd.concat([_read(root, "order_products__prior", usecols=cols),
                    _read(root, "order_products__train", usecols=cols)], ignore_index=True)
    op = op[op.order_id.isin(orders.order_id)]
    ev = op.merge(orders[["order_id", "user_id", "order_number", "t"]], on="order_id")

    cnt = ev.product_id.value_counts()
    ev = ev[ev.product_id.isin(cnt.index[cnt >= min_item_count])]
    n_ord = ev.groupby("user_id").order_number.nunique()
    ev = ev[ev.user_id.isin(n_ord.index[n_ord >= min_orders])]

    products = _read(root, "products").merge(_read(root, "aisles"), on="aisle_id") \
                                      .merge(_read(root, "departments"), on="department_id")
    pid = np.sort(ev.product_id.unique())
    item_of = pd.Series(np.arange(len(pid)), index=pid)
    ev = ev.assign(item=item_of[ev.product_id].to_numpy(),
                   user=ev.user_id.astype("category").cat.codes.to_numpy())
    ev["order"] = ev.groupby("user").order_number.rank(method="dense").astype(int) - 1
    events = ev.rename(columns={"add_to_cart_order": "pos"})[["user", "item", "order", "t", "pos"]] \
               .sort_values(["user", "order", "pos"]).reset_index(drop=True)
    items = products.set_index("product_id").loc[pid].reset_index()
    items = pd.DataFrame({"item": np.arange(len(pid)), "product_id": pid, "name": items.product_name.to_numpy(),
                          "aisle": items.aisle.to_numpy(), "department": items.department.to_numpy()})
    return events, items


# ----------------------------------------------------------------------------- фейковые данные

_ADJ = ["organic", "fresh", "low fat", "whole", "natural", "sparkling", "greek", "baby", "frozen", "roasted",
        "unsweetened", "vanilla", "spicy", "gluten free", "original", "family size", "mini", "classic"]


def make_fake_instacart(out_dir, n_users=3000, n_aisles=60, n_departments=12, products_per_aisle=40, seed=0):
    """
    Пишет CSV в формате Instacart. Структура похожа на настоящую: у пользователя любимые ряды (aisles),
    «свои» товары, которые он переказывает, и новые товары из любимых рядов; паузы между заказами 1–30 дней.
    Названия товаров собраны из слов ряда, так что текст несёт сигнал для семантических ID.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    aisle_names = [f"aisle{a} {w}" for a, w in zip(range(n_aisles), rng.choice(
        ["yogurt", "chips", "water", "bread", "cheese", "juice", "pasta", "soup", "tea", "coffee", "candy",
         "cereal", "eggs", "milk", "fruit", "vegetables", "frozen meals", "spices", "baby food", "pet food"], n_aisles))]
    aisle_dep = rng.integers(0, n_departments, n_aisles)
    rows, pid = [], 1
    for a in range(n_aisles):
        noun = aisle_names[a].split(" ", 1)[1]
        for _ in range(products_per_aisle):
            rows.append((pid, f"{' '.join(rng.choice(_ADJ, rng.integers(1, 3), replace=False))} {noun} {pid}",
                         a + 1, aisle_dep[a] + 1))
            pid += 1
    products = pd.DataFrame(rows, columns=["product_id", "product_name", "aisle_id", "department_id"])
    pop = rng.zipf(1.5, len(products)).clip(max=500).astype(float)
    by_aisle = [np.where(products.aisle_id.to_numpy() == a + 1)[0] for a in range(n_aisles)]

    orders, op, oid = [], [], 1
    for u in range(1, n_users + 1):
        fav_a = rng.choice(n_aisles, rng.integers(3, 9), replace=False)
        cand = np.concatenate([by_aisle[a] for a in fav_a])
        w = pop[cand] / pop[cand].sum()
        staples = set(rng.choice(cand, min(len(cand), rng.integers(3, 15)), replace=False, p=w).tolist())
        p_re = rng.uniform(0.3, 0.9)                      # насколько человек «рутинный»
        gap = rng.choice([3, 7, 14, 30])
        bought = set()
        for n in range(1, rng.integers(4, 40) + 1):
            orders.append((oid, u, "prior", n, None if n == 1 else int(min(30, max(0, rng.normal(gap, gap / 3))))))
            basket = [i for i in staples if rng.random() < p_re]
            basket += [i for i in bought if rng.random() < 0.05]
            basket += list(rng.choice(cand, rng.integers(1, 5), p=w))
            for k, i in enumerate(dict.fromkeys(basket)):
                op.append((oid, int(products.product_id[i]), k + 1, int(i in bought)))
                bought.add(i)
            oid += 1
    od = pd.DataFrame(orders, columns=["order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"])
    od["order_dow"], od["order_hour_of_day"] = 0, 12
    last = od.groupby("user_id").order_number.transform("max") == od.order_number
    od.loc[last, "eval_set"] = "train"
    opd = pd.DataFrame(op, columns=["order_id", "product_id", "add_to_cart_order", "reordered"])
    tr = opd.order_id.isin(od.order_id[last])
    od.to_csv(os.path.join(out_dir, "orders.csv"), index=False)
    opd[~tr].to_csv(os.path.join(out_dir, "order_products__prior.csv"), index=False)
    opd[tr].to_csv(os.path.join(out_dir, "order_products__train.csv"), index=False)
    products.to_csv(os.path.join(out_dir, "products.csv"), index=False)
    pd.DataFrame({"aisle_id": range(1, n_aisles + 1), "aisle": aisle_names}).to_csv(
        os.path.join(out_dir, "aisles.csv"), index=False)
    pd.DataFrame({"department_id": range(1, n_departments + 1),
                  "department": [f"department {d}" for d in range(n_departments)]}).to_csv(
        os.path.join(out_dir, "departments.csv"), index=False)
    return out_dir


# ----------------------------------------------------------------------------- протокол

@dataclass
class Split:
    hist: dict        # user -> (items, times, orders): история до целевого заказа (по порядку)
    targets: dict     # user -> np.array различных товаров целевого заказа
    t_query: dict     # user -> время целевого заказа (момент запроса: пользователь открыл приложение)

    @property
    def users(self):
        return np.array(sorted(self.targets))


def make_protocol(events):
    """Возвращает (train_events, val, test). train_events — всё до val-заказа каждого пользователя."""
    last = events.groupby("user")["order"].transform("max")
    train = events[events.order < last - 1]

    def build(target_off):
        tgt_order = last - target_off
        hist_ev = events[events.order < tgt_order]
        tgt_ev = events[events.order == tgt_order]
        hist = {u: (g.item.to_numpy(), g.t.to_numpy(), g.order.to_numpy()) for u, g in hist_ev.groupby("user")}
        targets = {u: np.unique(g.item.to_numpy()) for u, g in tgt_ev.groupby("user") if u in hist}
        t_query = tgt_ev.groupby("user").t.first().to_dict()
        return Split(hist, targets, t_query)

    return train.reset_index(drop=True), build(1), build(0)


def item_counts(train, n_items):
    return np.bincount(train["item"], minlength=n_items)
