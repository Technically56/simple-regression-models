"""Draw the report figures from the saved runs.

Run from the repository root:  python -m benchmark.figures [--lang en|tr]
Needs matplotlib (uv pip install matplotlib). Writes PNGs to report/figures/
(English) or report/figures/tr/ (Turkish).
Only reads files under runs/, so it never retrains or reloads the dataset.
Figure 6 needs runs/<final>/split_metrics.json (train/val/test scores).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "report" / "figures"

# Final model per algorithm, chosen by lowest validation cost (clipped at 5 sigma).
FINAL = {
    "logistic": "2026-11-18_032927_981410Z_logistic",  # lr 0.15
    "linear": "2026-11-18_034604_637472Z_linear",  # lr 0.008
}
# Same learning rate (0.001), without and with feature clipping.
CLIP_PAIR = {
    "Without clipping": "2026-11-18_031431_931331Z_linear",
    "Clipped at ±5σ": "2026-11-18_032308_783149Z_linear",
}

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
COLOR = {"logistic": BLUE, "linear": ORANGE}

TEXT = {
    "en": {
        "f0": "Figure 0. EMBER 2018 sample sizes: 1,000,000 raw samples, 800,000 used",
        "f0a": "Raw files → splits",
        "f0b": "Class balance per split",
        "unlabelled": "Unlabelled (removed)",
        "split_names": ["Train", "Validation", "Test"],
        "benign": "Benign (0)",
        "malware": "Malware (1)",
        "samples": "Samples",
        "logistic": "Logistic regression",
        "linear": "Linear regression",
        "f7": "Figure 7. Software architecture (UML class diagram)",
        "abstract": "«abstract»",
        "dataclass": "«dataclass»",
        "function": "«function»",
        "rt_note": ["used by main.py (CLI)", "and app.py (Streamlit)"],
        "rt_steps": ["builds model via MODEL_CLASSES", "trains with Trainer", "save_model(), evaluate()"],
        "norm_desc": ["(x − mean) · inv_std", "then clip to ±clip"],
        "eval_desc": ["accuracy, precision,", "recall, F1, confusion"],
        "leg_inherit": "inheritance (is a)",
        "leg_assoc": "association (uses / holds)",
        "leg_aggr": "aggregation (model holds backend)",
        "leg_dep": "dependency (calls)",
        "creates": "creates",
        "ce_short": "cross-entropy",
        "f1": "Figure 1. Confusion matrices on the same 200,000-sample test set",
        "acc": "acc",
        "of_row": "{p} of row",
        "pred": ["Predicted benign", "Predicted malware"],
        "actual": ["Actual benign", "Actual malware"],
        "f2": "Figure 2. Test metrics, same test data for both models",
        "metrics": ["Accuracy", "Precision", "Recall", "F1"],
        "score": "Score on test set",
        "f3": "Figure 3. Cost per epoch for the final runs (validation tracks training closely: no overfitting)",
        "train_cost": "Training cost",
        "val_cost": "Validation cost",
        "best": "best epoch {e}\nval J = {v}",
        "epoch": "Epoch",
        "cost": "Cost J ({c} + L2)",
        "ce": "cross-entropy",
        "mse": "½·MSE",
        "lr": "lr",
        "f4": "Figure 4. Linear regression at lr 0.001, without and with feature clipping",
        "no_clip": "Without clipping",
        "clip": "Clipped at ±5σ",
        "diverged": "diverged, early-stopped",
        "converging": "converging",
        "const": "always predict 0.5 (J = 0.125)  ",
        "val_log": "Validation cost J (log scale)",
        "f5": "Figure 5. Logistic regression learning-rate sweep (full data, λ = 1e-4)",
        "converged": "converged (early-stopped)",
        "limit": "hit epoch limit",
        "lr_axis": "Learning rate α (log scale)",
        "best_val": "Best validation cost J",
        "f6": "Figure 6. Accuracy per split: train ≈ validation (no overfitting), test lower (drift)",
        "splits": ["Train (540k)", "Validation (60k)", "Test (200k, newer files)"],
        "acc_axis": "Accuracy (axis starts at 80%)",
    },
    "tr": {
        "f0": "Şekil 0. EMBER 2018 örnek sayıları: 1.000.000 ham örnek, 800.000 kullanıldı",
        "f0a": "Ham dosyalar → kümeler",
        "f0b": "Kümelere göre sınıf dağılımı",
        "unlabelled": "Etiketsiz (çıkarıldı)",
        "split_names": ["Eğitim", "Doğrulama", "Test"],
        "benign": "Zararsız (0)",
        "malware": "Zararlı (1)",
        "samples": "Örnek sayısı",
        "logistic": "Lojistik regresyon",
        "linear": "Doğrusal regresyon",
        "f7": "Şekil 7. Yazılım mimarisi (UML sınıf diyagramı)",
        "abstract": "«soyut»",
        "dataclass": "«veri sınıfı»",
        "function": "«fonksiyon»",
        "rt_note": ["main.py (komut satırı) ve", "app.py (Streamlit) kullanır"],
        "rt_steps": ["MODEL_CLASSES ile model kurar", "Trainer ile eğitir", "save_model(), evaluate()"],
        "norm_desc": ["(x − ortalama) · inv_std,", "sonra ±clip ile kırpar"],
        "eval_desc": ["doğruluk, kesinlik,", "duyarlılık, F1, matris"],
        "leg_inherit": "kalıtım (bir türüdür)",
        "leg_assoc": "ilişki (kullanır / tutar)",
        "leg_aggr": "toplama (model arka ucu tutar)",
        "leg_dep": "bağımlılık (çağırır)",
        "creates": "oluşturur",
        "ce_short": "çapraz entropi",
        "f1": "Şekil 1. Aynı 200.000 örneklik test kümesinde karışıklık matrisleri",
        "acc": "doğruluk",
        "of_row": "{p} (satır oranı)",
        "pred": ["Tahmin: zararsız", "Tahmin: zararlı"],
        "actual": ["Gerçek: zararsız", "Gerçek: zararlı"],
        "f2": "Şekil 2. Test metrikleri (iki model için aynı test verisi)",
        "metrics": ["Doğruluk", "Kesinlik", "Duyarlılık", "F1"],
        "score": "Test kümesindeki skor",
        "f3": "Şekil 3. Nihai modellerde epoch başına maliyet (doğrulama eğitimi yakından izliyor: aşırı öğrenme yok)",
        "train_cost": "Eğitim maliyeti",
        "val_cost": "Doğrulama maliyeti",
        "best": "en iyi epoch {e}\ndoğrulama J = {v}",
        "epoch": "Epoch",
        "cost": "Maliyet J ({c} + L2)",
        "ce": "çapraz entropi",
        "mse": "½·MSE",
        "lr": "α",
        "f4": "Şekil 4. Doğrusal regresyon (α = 0,001): öznitelik kırpma olmadan ve kırpma ile",
        "no_clip": "Kırpma yok",
        "clip": "±5σ ile kırpılmış",
        "diverged": "ıraksadı, erken durduruldu",
        "converging": "yakınsıyor",
        "const": "her zaman 0,5 tahmini (J = 0,125)  ",
        "val_log": "Doğrulama maliyeti J (log ölçek)",
        "f5": "Şekil 5. Lojistik regresyon öğrenme oranı taraması (tüm veri, λ = 1e-4)",
        "converged": "yakınsadı (erken durdu)",
        "limit": "epoch sınırına ulaştı",
        "lr_axis": "Öğrenme oranı α (log ölçek)",
        "best_val": "En iyi doğrulama maliyeti J",
        "f6": "Şekil 6. Kümelere göre doğruluk: eğitim ≈ doğrulama (aşırı öğrenme yok), test daha düşük (zamansal kayma)",
        "splits": ["Eğitim (540 bin)", "Doğrulama (60 bin)", "Test (200 bin, daha yeni dosyalar)"],
        "acc_axis": "Doğruluk (eksen %80'den başlar)",
    },
}
LANG = "en"


def T(key):
    return TEXT[LANG][key]


def num(x, decimals):
    """Decimal number in the report language (Turkish uses a decimal comma)."""
    s = f"{x:,.{decimals}f}"
    return s.translate(str.maketrans(",.", ".,")) if LANG == "tr" else s


def pct(x, decimals=1):
    s = num(100 * x, decimals)
    return f"%{s}" if LANG == "tr" else f"{s}%"


def count(n):
    return num(n, 0)


def decimal_axis(ax, decimals):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: num(v, decimals)))
INK, INK_2, GRID, SURFACE = "#1f1f1d", "#5e5d59", "#e4e3df", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def load_run(name):
    d = RUNS / name
    events = [json.loads(line) for line in (d / "metrics.jsonl").open(encoding="utf-8")]
    run = {"dir": d, "name": name, "events": events}
    run["start"] = next((e for e in events if e["event"] == "run"), {})
    run["epochs"] = [e for e in events if e["event"] == "epoch"]
    run["early_stop"] = any(e["event"] == "early_stop" for e in events)
    with np.load(d / "model.npz") as f:
        run["cfg"] = json.loads(str(f["config"]))
    run["cfg"].setdefault("clip", None)
    for key, file in (("test", "test_metrics.json"), ("splits", "split_metrics.json")):
        p = d / file
        run[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return run


def all_full_runs():
    runs = []
    for d in sorted(RUNS.iterdir()):
        try:
            r = load_run(d.name)
        except (OSError, KeyError, ValueError):
            continue  # unfinished run
        if r["start"].get("train_rows") == 540_000 and r["epochs"] and r["test"]:
            runs.append(r)
    return runs


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# Counted from dataset/ember2018/*.jsonl (label -1 = unlabelled) and the processed y_*.npy.
RAW_TRAIN, RAW_TEST, UNLABELLED = 800_000, 200_000, 200_000
CLASS_COUNTS = {  # split: (benign, malware)
    "train": (270_240, 269_760),
    "val": (29_760, 30_240),
    "test": (100_000, 100_000),
}
SPLIT_COLOR = {"train": "#1baf7a", "val": "#eda100", "test": "#e87ba4"}


def fig0_dataset():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.6), height_ratios=[1, 1.6])
    names = dict(zip(["train", "val", "test"], T("split_names")))
    # Panel A: one bar of all 1,000,000 raw samples.
    parts = [(names[k], sum(CLASS_COUNTS[k]), SPLIT_COLOR[k]) for k in ("train", "val", "test")]
    parts.append((T("unlabelled"), UNLABELLED, GRAY))
    left = 0
    for label, n, color in parts:
        ax1.barh(0, n, left=left, color=color, edgecolor=SURFACE, linewidth=2,
                 hatch="//" if color == GRAY else None)
        text = f"{label}\n{count(n)}"
        if n < 100_000:  # too narrow for the label: put it above the bar
            ax1.text(left + n / 2, 0.45, text, ha="center", va="bottom", fontsize=9, color=INK)
        else:
            box = dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none") if color == GRAY else None
            ax1.text(left + n / 2, 0, text, ha="center", va="center", fontsize=9, color=INK, bbox=box)
        left += n
    ax1.set_xlim(0, RAW_TRAIN + RAW_TEST)
    ax1.set_ylim(-0.5, 1.0)
    ax1.set_yticks([])
    ax1.set_title(T("f0a"))
    ax1.grid(False)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: count(v)))
    for sp in ("left", "bottom"):
        ax1.spines[sp].set_visible(False)
    ax1.tick_params(axis="x", length=0, labelbottom=False)
    # Panel B: benign vs malware per split.
    keys = ["train", "val", "test"]
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        b, m = CLASS_COUNTS[k]
        ax2.barh(yi, b, color=BLUES[2], edgecolor=SURFACE, linewidth=2, label=T("benign") if k == "train" else None)
        ax2.barh(yi, m, left=b, color=BLUES[5], edgecolor=SURFACE, linewidth=2, label=T("malware") if k == "train" else None)
        ax2.text(b / 2, yi, count(b), ha="center", va="center", fontsize=9, color=INK)
        ax2.text(b + m / 2, yi, count(m), ha="center", va="center", fontsize=9, color="white")
    ax2.set_yticks(y, [names[k] for k in keys])
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel(T("samples"))
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: count(v)))
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="lower right")
    ax2.set_title(T("f0b"))
    fig.suptitle(T("f0"), x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig0_dataset_sizes.png")


def fig7_uml():
    from matplotlib.patches import Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(17.5, 9.5))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    line_h = 0.27
    boxes = {}

    def cls(key, x, y_top, name, stereo=None, attrs=(), methods=(), width=2.9, fill="#eef4fc"):
        head = [stereo, name] if stereo else [name]
        sections = [head] + [list(sec) for sec in (attrs, methods) if sec]
        heights = [len(sec) * line_h + 0.16 for sec in sections]
        h = sum(heights)
        left = x - width / 2
        ax.add_patch(Rectangle((left, y_top - h), width, h, fc=SURFACE, ec=INK, lw=1.2, zorder=2))
        ax.add_patch(Rectangle((left, y_top - heights[0]), width, heights[0], fc=fill, ec=INK, lw=1.2, zorder=2))
        y = y_top
        for i, (sec, hh) in enumerate(zip(sections, heights)):
            for j, text in enumerate(sec):
                ty = y - 0.08 - (j + 0.5) * line_h
                if i == 0:
                    is_name = j == len(sec) - 1
                    ax.text(x, ty, text, ha="center", va="center", zorder=3, color=INK,
                            fontsize=10.5 if is_name else 8.5, fontweight="bold" if is_name else "normal",
                            fontstyle="italic" if is_name and stereo == T("abstract") else "normal")
                else:
                    ax.text(left + 0.12, ty, text, ha="left", va="center", fontsize=8.5,
                            family="monospace", color=INK, zorder=3)
            y -= hh
            if i < len(sections) - 1:
                ax.plot([left, left + width], [y, y], color=INK, lw=1, zorder=3)
        boxes[key] = dict(x=x, top=y_top, bottom=y_top - h, left=left, right=left + width,
                          mid=y_top - h / 2)

    def arrow(start, end, kind, label=None, rad=0.0):
        style = dict(color=INK, lw=1.2, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}")
        if kind == "inherit":
            style.update(arrowstyle="-|>,head_length=0.6,head_width=0.35", fc=SURFACE)
        elif kind == "assoc":
            style.update(arrowstyle="->,head_length=0.5,head_width=0.25")
        elif kind == "dep":
            style.update(arrowstyle="->,head_length=0.5,head_width=0.25", linestyle=(0, (4, 3)))
        else:  # aggregation: plain line, hollow diamond drawn at the owner (end)
            style.update(arrowstyle="-")
        ax.annotate("", xy=end, xytext=start, arrowprops=style, zorder=1)
        if kind == "aggr":
            (sx, sy), (ex, ey) = start, end
            d = np.hypot(ex - sx, ey - sy)
            ux, uy = (sx - ex) / d, (sy - ey) / d
            px, py = -uy, ux
            size, half_w = 0.34, 0.12
            pts = [(ex, ey), (ex + ux * size / 2 + px * half_w, ey + uy * size / 2 + py * half_w),
                   (ex + ux * size, ey + uy * size), (ex + ux * size / 2 - px * half_w, ey + uy * size / 2 - py * half_w)]
            ax.add_patch(Polygon(pts, closed=True, fc=SURFACE, ec=INK, lw=1.2, zorder=4))
        if label:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(mx, my + 0.1, label, ha="center", va="bottom", fontsize=8.5, color=INK_2, zorder=3)

    col_a, col_b, col_c, col_d = 2.0, 7.0, 12.0, 16.4
    func_fill = "#f3f2ee"
    # Row 1: orchestration
    cls("run", col_a, 10.2, "main.run_training()", T("function"), T("rt_note"), T("rt_steps"), width=3.4, fill=func_fill)
    cls("trainer", col_b, 10.2, "Trainer", None, ["model, backend", "train_data, val_data, cfg"],
        ["train(on_epoch) → history", "_prepare(x, y)", "_validate() → val J"], width=3.4)
    cls("cfg", col_c, 10.2, "TrainerConfig", T("dataclass"),
        ["model, lr, l2, epochs", "batch_size, patience", "seed, clip"])
    cls("norm", col_d, 10.2, "normalise()", T("function"), T("norm_desc"), width=2.6, fill=func_fill)
    # Row 2: abstract base classes
    cls("bbackend", col_a, 6.6, "BaseBackend", T("abstract"), ["name, is_gpu"],
        ["to_device(x), to_host(x)", "matmul(a, b), sigmoid(a)", "clip(a, lo, hi)", "axpy_(α, x, y)"], width=3.2)
    cls("bmodel", col_b, 6.6, "BaseModel", T("abstract"), ["w, b, l2, be"],
        ["forward(x) → h(x) {abstract}", "cost(h, y) → J {abstract}", "gradient(x, y) → J, grads",
         "step(grads, lr)", "predict(x, 0.5)", "state(), load(state)"], width=3.6)
    cls("bdata", col_c, 6.6, "BaseDataset", T("abstract"), ["x, y, mean, inv_std"],
        ["load(split)", "batches(size, rng)", "subset(n)"])
    cls("eval", col_d, 6.6, "evaluate()", T("function"), T("eval_desc"), width=2.6, fill=func_fill)
    # Row 3: concrete classes
    cls("cpu", 1.05, 2.6, "CPUBackend", None, (), ["NumPy"], width=1.9)
    cls("gpu", 3.05, 2.6, "GPUBackend", None, (), ["CuPy"], width=1.9)
    cls("lin", 5.55, 2.6, "LinearRegressionModel", None, (), ["h = w·x + b", "J = ½·MSE + ½λ‖w‖²"], width=3.0)
    cls("log", 8.85, 2.6, "LogisticRegressionModel", None, (),
        ["h = σ(w·x + b)", f"J = {T('ce_short')} + ½λ‖w‖²"], width=3.4)
    cls("ember", 12.3, 2.6, "EmberDataset", None, (), ["build(limit, seed, workers)"])

    for child, parent in (("cpu", "bbackend"), ("gpu", "bbackend"), ("lin", "bmodel"),
                          ("log", "bmodel"), ("ember", "bdata")):
        c, pb = boxes[child], boxes[parent]
        tx = min(max(c["x"], pb["left"] + 0.4), pb["right"] - 0.4)
        arrow((c["x"], c["top"]), (tx, pb["bottom"]), "inherit")

    tr, bm, bb, bd = boxes["trainer"], boxes["bmodel"], boxes["bbackend"], boxes["bdata"]
    arrow((bb["right"], bb["mid"]), (bm["left"], bb["mid"]), "aggr")
    arrow((tr["x"], tr["bottom"]), (bm["x"], bm["top"]), "assoc")
    arrow((tr["right"] - 0.3, tr["bottom"]), (bd["x"], bd["top"]), "assoc", rad=0.2)
    arrow((tr["left"] + 0.3, tr["bottom"]), (bb["x"], bb["top"]), "assoc", rad=-0.2)
    arrow((tr["right"], tr["top"] - 0.55), (boxes["cfg"]["left"], tr["top"] - 0.55), "assoc")
    arrow((boxes["run"]["right"], tr["top"] - 0.55), (tr["left"], tr["top"] - 0.55), "dep", T("creates"))
    arrow((boxes["eval"]["x"], boxes["eval"]["top"]), (boxes["norm"]["x"], boxes["norm"]["bottom"]), "dep")
    # evaluate() -> BaseModel: elbow route below BaseDataset so it crosses no box
    ev = boxes["eval"]
    route_y = bd["bottom"] - 0.35
    ax.plot([ev["x"], ev["x"], bd["right"] + 0.2, bd["right"] + 0.2], [ev["bottom"], route_y, route_y, route_y],
            color=INK, lw=1.2, linestyle=(0, (4, 3)), zorder=1)
    arrow((bd["right"] + 0.2, route_y), (bm["right"], route_y), "dep")

    # Legend
    lx, ly = 15.1, 2.9
    for i, (kind, key) in enumerate((("inherit", "leg_inherit"), ("assoc", "leg_assoc"),
                                     ("aggr", "leg_aggr"), ("dep", "leg_dep"))):
        y = ly - i * 0.5
        if kind == "aggr":
            arrow((lx + 0.7, y), (lx, y), kind)
        else:
            arrow((lx, y), (lx + 0.7, y), kind)
        ax.text(lx + 0.85, y, T(key), va="center", fontsize=8.5, color=INK_2)
    ax.set_title(T("f7"), loc="left", fontsize=12)
    save(fig, "fig7_uml_architecture.png")


def fig1_confusion(final):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    vmax = max(max(map(max, r["test"]["confusion"])) for r in final.values())
    for ax, (model, r) in zip(axes, final.items()):
        cm = np.array(r["test"]["confusion"])
        ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUES), vmin=0, vmax=vmax)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / cm[i].sum()
                dark = cm[i, j] > vmax * 0.55
                ax.text(j, i, count(cm[i, j]) + "\n" + T("of_row").format(p=pct(share)), ha="center", va="center",
                        color="white" if dark else INK, fontsize=10)
        ax.set_xticks([0, 1], T("pred"))
        ax.set_yticks([0, 1], T("actual"))
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{T(model)}  ({T('acc')} {pct(r['test']['accuracy'])})")
    fig.suptitle(T("f1"),
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_confusion_matrices.png")


def fig2_metrics(final):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = T("metrics")
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, (model, r) in enumerate(final.items()):
        vals = [r["test"][m] for m in metrics]
        bars = ax.bar(x + (k - 0.5) * (width + 0.02), vals, width, color=COLOR[model], label=T(model))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, num(v, 3), ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(T("score"))
    decimal_axis(ax, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 1.12))
    ax.set_title(T("f2"), pad=28)
    save(fig, "fig2_metric_comparison.png")


def fig3_cost(final):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (model, r) in zip(axes, final.items()):
        ep = [e["epoch"] for e in r["epochs"]]
        tr = [e["train_J"] for e in r["epochs"]]
        va = [e["val_J"] for e in r["epochs"]]
        ax.plot(ep, tr, color=COLOR[model], label=T("train_cost"))
        ax.plot(ep, va, color=COLOR[model], linestyle="--", label=T("val_cost"))
        best = int(np.argmin(va))
        ax.plot(ep[best], va[best], "o", ms=7, color=COLOR[model], mec=SURFACE, mew=2)
        ax.annotate(T("best").format(e=ep[best], v=num(va[best], 4)), (ep[best], va[best]),
                    xytext=(-10, 28), textcoords="offset points", ha="right", fontsize=9, color=INK_2)
        cost = T("mse") if model == "linear" else T("ce")
        lr = num(r["cfg"]["lr"], 3).rstrip("0")
        ax.set_title(f"{T(model)} ({T('lr')} = {lr})")
        ax.set_xlabel(T("epoch"))
        ax.set_ylabel(T("cost").format(c=cost))
        decimal_axis(ax, 2)
        ax.legend(loc="upper right")
    fig.suptitle(T("f3"),
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_cost_curves.png")


def fig4_clipping():
    fig, ax = plt.subplots(figsize=(8, 4))
    styles = {"Without clipping": (GRAY, "--"), "Clipped at ±5σ": (ORANGE, "-")}
    names = {"Without clipping": T("no_clip"), "Clipped at ±5σ": T("clip")}
    for key, name in CLIP_PAIR.items():
        label = names[key]
        r = load_run(name)
        ep = [e["epoch"] for e in r["epochs"]]
        va = [e["val_J"] for e in r["epochs"]]
        color, ls = styles[key]
        ax.plot(ep, va, color=color, linestyle=ls, marker="o" if len(ep) < 10 else None, ms=6, label=label)
        end = T("diverged") if key == "Without clipping" else T("converging")
        ax.annotate(f"{label}: {end}", (ep[-1], va[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    ax.axhline(0.125, color=INK_2, linewidth=1, linestyle=":")
    ax.text(ax.get_xlim()[1], 0.125, T("const"), ha="right", va="bottom",
            fontsize=9, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlabel(T("epoch"))
    ax.set_ylabel(T("val_log"))
    ax.legend(loc="upper right")
    ax.set_title(T("f4"))
    save(fig, "fig4_clipping_linear.png")


def fig5_lr_sweep(runs):
    fig, ax = plt.subplots(figsize=(8, 4))
    for clipped, color, label in ((False, GRAY, T("no_clip")), (True, BLUE, T("clip"))):
        best = {}
        for r in runs:
            if r["cfg"]["model"] != "logistic" or bool(r["cfg"]["clip"]) != clipped:
                continue
            lr = r["cfg"]["lr"]
            if lr not in best or len(r["epochs"]) > len(best[lr]["epochs"]):
                best[lr] = r  # longest run per learning rate
        lrs = sorted(best)
        vals = [min(e["val_J"] for e in best[lr]["epochs"]) for lr in lrs]
        done = [best[lr]["early_stop"] for lr in lrs]
        ax.plot(lrs, vals, color=color, linewidth=1.5, label=label)
        ax.scatter([l for l, d in zip(lrs, done) if d], [v for v, d in zip(vals, done) if d],
                   s=40, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter([l for l, d in zip(lrs, done) if not d], [v for v, d in zip(vals, done) if not d],
                   s=40, facecolor=SURFACE, edgecolor=color, linewidth=1.5, zorder=3)
    ax.scatter([], [], s=40, color=INK_2, label=T("converged"))
    ax.scatter([], [], s=40, facecolor=SURFACE, edgecolor=INK_2, label=T("limit"))
    ax.set_xscale("log")
    ax.set_xlabel(T("lr_axis"))
    ax.set_ylabel(T("best_val"))
    decimal_axis(ax, 2)
    ax.legend(loc="upper center", ncols=2)
    ax.set_title(T("f5"))
    save(fig, "fig5_lr_sweep_logistic.png")


def fig6_splits(final):
    splits = ["train", "val", "test"]
    labels = T("splits")
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, r in final.items():
        if not r["splits"]:
            print("skip figure 6: no split_metrics.json in", r["name"])
            plt.close(fig)
            return
        acc = [r["splits"][s]["accuracy"] for s in splits]
        ax.plot(range(3), acc, color=COLOR[model], marker="o", ms=8, mec=SURFACE, mew=2, label=T(model))
        for i, a in enumerate(acc):
            below = model == "linear"
            ax.annotate(pct(a), (i, a), xytext=(0, -16 if below else 9), textcoords="offset points", ha="center",
                        fontsize=9, color=INK)
    ax.set_xticks(range(3), labels)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0.80, 0.95)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v, 0)))
    ax.set_ylabel(T("acc_axis"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    ax.set_title(T("f6"))
    save(fig, "fig6_split_accuracy.png")


def main():
    global LANG, OUT
    parser = argparse.ArgumentParser(description="Draw the report figures from runs/.")
    parser.add_argument("--lang", choices=sorted(TEXT), default="en")
    LANG = parser.parse_args().lang
    if LANG != "en":
        OUT = OUT / LANG
    runs = all_full_runs()
    final = {m: load_run(name) for m, name in FINAL.items()}
    fig0_dataset()
    fig1_confusion(final)
    fig2_metrics(final)
    fig3_cost(final)
    fig4_clipping()
    fig5_lr_sweep(runs)
    fig6_splits(final)
    fig7_uml()


if __name__ == "__main__":
    main()
