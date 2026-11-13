"""
Собирает самодостаточный ноутбук для Kaggle: исходники candgen/ вставляются в ячейки %%writefile.

  python kaggle/build_notebook.py                                  # → kaggle/candgen_kaggle.ipynb
  python kaggle/build_notebook.py --quick --no-gpu --out <path>    # дымовая версия для локальной проверки
"""
import argparse
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["instacart.py", "semantic_ids.py", "models.py", "run.py"]

INTRO = """# Кандген на Instacart: семантические ID vs dense vs история + переключатель источников

Ноутбук самодостаточный: код записывается в файлы ячейками `%%writefile`.

**Как запустить на Kaggle**

1. *File → Import Notebook* — загрузи этот `.ipynb`.
2. Справа *Add Input* → вкладка *Competitions* → **Instacart Market Basket Analysis**
   (нужно один раз принять правила соревнования на его странице). Подойдёт и любой датасет-зеркало
   с теми же файлами (`orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`,
   `aisles.csv`, `departments.csv`, можно `.csv.zip`) — ноутбук сам найдёт их в `/kaggle/input`.
3. *Settings → Accelerator → GPU T4 x2* (или P100). Интернет не нужен.
4. Проверка: `QUICK = True` → *Run All* (фейковые данные, пара минут).
   Полный прогон: `QUICK = False` → *Save Version → Save & Run All (Commit)*.
5. Результаты: `results.zip` во вкладке *Output* (`summary.md`, `sources.csv`, `union.csv`, `latency.csv`, `log.json`, лог).

Оценка времени на T4 при 30 тыс. пользователей: ~30–60 мин (не замер).
"""

SETUP = """import os, sys, time, subprocess, torch

REQUIRE_GPU = {require_gpu}
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет GPU")
if REQUIRE_GPU:
    assert torch.cuda.is_available(), "Включи GPU: Settings → Accelerator"

BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
WORK = os.path.join(BASE, "candgen")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
print("работаем в", WORK)
try:
    import tabulate  # noqa: F401
except ImportError:
    os.system(f"{{sys.executable}} -m pip install -q tabulate")


def run(args, log):
    \"\"\"Запуск скрипта: вывод по ходу + лог (кроссплатформенный аналог `| tee`).\"\"\"
    print("$ python", args, flush=True)
    env = {{**os.environ, "PYTHONIOENCODING": "utf-8"}}
    with open(log, "w", encoding="utf-8") as f, subprocess.Popen(
            f'"{{sys.executable}}" -u {{args}}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", env=env) as p:
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
    assert p.returncode == 0, f"процесс упал (код {{p.returncode}}), см. {{log}}"
"""

CONFIG = """# ---- настройки ----
QUICK = {quick}              # True — фейковые данные формата Instacart, пара минут
N_USERS = 30000              # подвыборка пользователей Instacart (всего ~206 тыс.)
SID_CODES = "128 128 128"    # размеры кодбуков residual k-means по уровням
SID_SOURCES = "text cf"      # text — SID из текста товара, cf — из эмбеддингов dense-модели
KMAX, KS = 200, "10 50 100 200"
COMMON = f"--n-users {{N_USERS}} --sid-codes {{SID_CODES}} --sid-sources {{SID_SOURCES}} --kmax {{KMAX}} --ks {{KS}}"
if QUICK:
    COMMON = ("--fake --fake-users 1200 --n-users 1200 --min-item-count 5 --max-epochs 2 --per-user 3 "
              "--val-users 500 --sid-codes 32 32 32 --kmax 100 --ks 10 50 100")
os.makedirs("results", exist_ok=True)
print(COMMON)
"""

RUN = """t0 = time.time()
run(f"run.py {COMMON} --out results/main", "results/main.log")
print(f"готово за {(time.time() - t0) / 60:.1f} мин")
"""

SHOW = """from IPython.display import Markdown, display
display(Markdown(open("results/main/summary.md", encoding="utf-8").read()))
"""

ZIP = """import shutil
for junk in ("results/main/fake_data",):
    shutil.rmtree(junk, ignore_errors=True)
print("сохранено:", shutil.make_archive(os.path.join(BASE, "results"), "zip", "results"))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "kaggle", "candgen_kaggle.ipynb"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETUP.format(require_gpu=not a.no_gpu)),
             nbf.v4.new_markdown_cell("## Код\nСгенерировано `kaggle/build_notebook.py` из исходников `candgen/`.")]
    for name in SOURCES:
        with open(os.path.join(ROOT, name), encoding="utf-8") as f:
            cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + f.read()))
    cells += [nbf.v4.new_markdown_cell("## Прогон"), nbf.v4.new_code_cell(CONFIG.format(quick=a.quick)),
              nbf.v4.new_code_cell(RUN), nbf.v4.new_markdown_cell("## Результаты"),
              nbf.v4.new_code_cell(SHOW), nbf.v4.new_code_cell(ZIP)]
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"},
                      "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": False}}
    nbf.validate(nb)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    nbf.write(nb, a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()
