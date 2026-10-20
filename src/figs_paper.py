"""Publication figures for the JBHI paper (vector PDF, sized to the IEEE column/page width)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
COLW, PAGEW = 3.5, 7.16
HL = "#c2185b"
COL = {"Logistic regression": "#1f77b4", "Grid-tuned KELM": "#9467bd", "TabTransformer": "#7f7f7f",
       "Tabular Mamba": "#9e9d24", "Stacked DL": "#8c564b", "Random forest": "#ff7f0e", "CM-KELM": HL,
       "LightGBM": "#17becf", "XGBoost": "#2ca02c", "JIT Transformer": "#3949ab"}


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.2g}" if p < 0.01 else f"{p:.2f}"


# ---------- Fig. S2: discrimination with CIs (eos2 test set) ----------
M = [("Logistic regression", .8621, .8492, .8752, .2792, .2517, .3115),
     ("Tabular Mamba", .8750, .8628, .8873, .2941, .2669, .3256),
     ("Grid-tuned KELM", .8757, .8632, .8876, .2960, .2687, .3262),
     ("TabTransformer", .8806, .8685, .8924, .3074, .2807, .3399),
     ("Stacked DL", .8808, .8687, .8925, .3192, .2911, .3527),
     ("Random forest", .8830, .8709, .8948, .3356, .3070, .3693),
     ("CM-KELM", .8860, .8740, .8970, .3419, .3122, .3757),
     ("JIT Transformer", .8903, .8789, .9015, .3427, .3126, .3752),
     ("XGBoost", .8923, .8809, .9032, .4026, .3696, .4362),
     ("LightGBM", .8943, .8825, .9050, .3920, .3589, .4269)]
fig, axs = plt.subplots(1, 2, figsize=(PAGEW, 2.6), sharey=True)
for k, (ax, c0, title) in enumerate(zip(axs, (1, 4), ("AUROC (95% CI)", "AUPRC (95% CI)"))):
    for i, r in enumerate(M):
        c, hl = COL[r[0]], r[0] == "CM-KELM"
        ax.errorbar(r[c0], i, xerr=[[r[c0] - r[c0 + 1]], [r[c0 + 2] - r[c0]]], fmt="o", color=c,
                    ms=5 if hl else 3.8, lw=1.6 if hl else 1.0, capsize=2)
        ax.text(r[c0 + 2] + (0.002 if k == 0 else 0.005), i, f"{r[c0]:.3f}", va="center", fontsize=7,
                color=c, fontweight="bold" if hl else "normal")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.axhspan(5.5, 6.5, color=HL, alpha=0.07, lw=0)
axs[0].set_yticks(range(len(M)), [r[0] for r in M])
axs[0].set_xlim(0.84, 0.92)
axs[1].set_xlim(0.22, 0.47)
fig.tight_layout()
fig.savefig("f1_discrimination.pdf")

# ---------- Fig. 2: paired differences vs grid-tuned KELM (Holm p) ----------
D = [("Logistic regression", -.0135, -.0205, -.0066, 6.79e-4, -.0168, -.0428, .0086, 0.591),
     ("Tabular Mamba", -.0007, -.0076, .0062, 0.842, -.0019, -.0263, .0226, 0.829),
     ("TabTransformer", .0049, -.0008, .0106, 0.261, .0114, -.0116, .0337, 0.591),
     ("Stacked DL", .0051, -.0007, .0109, 0.261, .0232, .0004, .0468, 0.184),
     ("Random forest", .0073, .0006, .0140, 0.128, .0397, .0120, .0664, 0.005),
     ("CM-KELM", .0103, .0060, .0146, 1.34e-5, .0459, .0265, .0662, 0.0045),
     ("JIT Transformer", .0147, .0087, .0206, 1.07e-5, .0467, .0203, .0720, 0.0045),
     ("XGBoost", .0166, .0103, .0230, 2.31e-6, .1066, .0796, .1318, 0.0045),
     ("LightGBM", .0186, .0121, .0250, 1.53e-7, .0961, .0679, .1213, 0.0045)]
fig, axs = plt.subplots(1, 2, figsize=(PAGEW, 2.5), sharey=True)
for k, (ax, o, title) in enumerate(zip(axs, (1, 5), ("ΔAUROC (95% CI)", "ΔAUPRC (95% CI)"))):
    for i, r in enumerate(D):
        c, hl = COL[r[0]], r[0] == "CM-KELM"
        ax.errorbar(r[o], i, xerr=[[r[o] - r[o + 1]], [r[o + 2] - r[o]]], fmt="s", color=c,
                    ms=4.5 if hl else 3.5, lw=1.6 if hl else 1.0, capsize=2)
        ax.text(1.01, i, f"p {fmt_p(r[o + 3])}" if fmt_p(r[o + 3]).startswith("<") else f"p = {fmt_p(r[o + 3])}",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.5, color=c,
                fontweight="bold" if hl else "normal")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.axhspan(4.5, 5.5, color=HL, alpha=0.07, lw=0)
axs[0].set_yticks(range(len(D)), [r[0] for r in D])
fig.tight_layout(w_pad=5)
fig.savefig("f2_forest.pdf")

# ---------- Fig. 3: preoperative vs end of surgery (single column) ----------
fig, axs = plt.subplots(1, 2, figsize=(COLW, 2.3))
lab = ["Preop.\n(37 var.)", "End of\nsurgery\n(93 var.)"]
data = {"AUROC": {"Grid-tuned KELM": (.871, .876), "CM-KELM": (.877, .886), "XGBoost": (.878, .892)},
        "AUPRC": {"Grid-tuned KELM": (.308, .296), "CM-KELM": (.314, .342), "XGBoost": (.343, .403)}}
for ax, (met, d) in zip(axs, data.items()):
    w = 0.27
    for j, (m, v) in enumerate(d.items()):
        xs = np.arange(2) + (j - 1) * w
        ax.bar(xs, v, w, color=COL[m], label=m, alpha=0.95 if m == "CM-KELM" else 0.75)
        for x, val in zip(xs, v):
            ax.text(x, val, f"{val:.3f}".lstrip("0"), ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(range(2), lab, fontsize=6.5)
    ax.set_title(met)
    ax.set_ylim((0.83, 0.90) if met == "AUROC" else (0.2, 0.43))
    ax.grid(axis="y", alpha=0.3)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=6.5)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
fig.savefig("f3_preop_eos.pdf")

# ---------- Fig. 5: selection frequency (single column) ----------
F = [("Age", 10, 0), ("Creatinine", 10, 0), ("Urea nitrogen", 10, 0), ("Urological surgery", 10, 0),
     ("Anesthesia duration", 9, 0), ("ASA class", 9, 0), ("Arterial line", 9, 1), ("Urine output, mL/kg/h", 9, 1),
     ("Emergency surgery", 7, 0), ("Platelets", 7, 0), ("Cardiothoracic surgery", 7, 0),
     ("Internal medicine", 7, 0), ("Urine output, total", 7, 1)]
fig, ax = plt.subplots(figsize=(COLW, 2.7))
yy = np.arange(len(F))[::-1]
ax.barh(yy, [f[1] for f in F], color=[HL if f[2] else "#5c6bc0" for f in F], height=0.65)
ax.set_yticks(yy, [f[0] for f in F])
ax.set_xlim(0, 10.8)
ax.set_xticks(range(0, 11, 2))
ax.set_xlabel("runs selecting the variable (of 10)", fontsize=7)
for v, f in zip(yy, F):
    ax.text(f[1] + 0.12, v, str(f[1]), va="center", fontsize=6.5)
fig.legend(handles=[Patch(color="#5c6bc0", label="preoperative"), Patch(color=HL, label="intraoperative")],
           frameon=False, loc="upper center", ncol=2, fontsize=6.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("f5_features.pdf")

# ---------- Fig. S1: attrition ----------
fig, ax = plt.subplots(figsize=(PAGEW, 3.0))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10.7)
main = [("INSPIRE: 130,960 operations\n99,886 patients", 9.1), ("129,594 operations", 6.9), ("64,667 operations", 4.7),
        ("51,445 patients", 2.5), ("Final cohort: 50,484 patients\n2,465 AKI (4.9%)", 0.4)]
exc = ["no preoperative creatinine: 1,366", "no postoperative creatinine within 7 days: 64,927",
       "repeat operations (first eligible kept): 13,222",
       "pre-existing kidney failure (top-coded creatinine or Z49): 961"]
for i, (t, y) in enumerate(main):
    last = i == len(main) - 1
    two = "\n" in t
    ax.add_patch(FancyBboxPatch((0.6, y - 0.05), 3.6, 1.15 if two else 0.8, boxstyle="round,pad=0.05",
                                fc="#fbe9f0" if last else "white", ec=HL if last else "#444", lw=1))
    ax.text(2.4, y + (0.52 if two else 0.35), t, ha="center", va="center", fontsize=8,
            fontweight="bold" if last else "normal")
for i, t in enumerate(exc):
    y0 = main[i][1]; y1 = main[i + 1][1] + (1.15 if "\n" in main[i + 1][0] else 0.8)
    ax.annotate("", xy=(2.4, y1 + 0.05), xytext=(2.4, y0 - 0.07), arrowprops=dict(arrowstyle="->", lw=0.9))
    ym = (y0 + y1) / 2
    ax.plot([2.4, 4.75], [ym, ym], color="#444", lw=0.8)
    ax.add_patch(FancyBboxPatch((4.75, ym - 0.33), 5.2, 0.66, boxstyle="round,pad=0.04", fc="#f5f5f5", ec="#999", lw=0.8))
    ax.text(7.35, ym, "Excluded, " + t, ha="center", va="center", fontsize=6.6)
ax.text(7.35, 0.75, "Stratified 70/30 split\nTraining 35,338 (1,725 AKI) · Test 15,146 (740 AKI)", ha="center",
        va="center", fontsize=7.5, color=HL)
fig.tight_layout()
fig.savefig("f0_attrition.pdf")

# ---------- Fig. S3: multimeme evidence ----------
fig, ax = plt.subplots(figsize=(PAGEW, 2.2))
rows = [("Primary: fuzzy + credit vs static\nseeds 1–10 · island-ensemble AUPRC", 0.002, "p = 0.56", "#9e9e9e"),
        ("Discovery: tabu/GLS meme vs static\nseeds 1–10 · single-run AUPRC", 0.012, "p = 0.004", HL),
        ("Replication: tabu/GLS\nseeds 11–20 · single-run AUPRC", 0.004, "p = 0.77", "#f48fb1"),
        ("Pre-specified confirmation: tabu/GLS\nseeds 21–50 · single-run AUPRC", 0.0029, "p = 0.24", "#f48fb1"),
        ("Exploratory: tabu/GLS\nseeds 21–50 · island-ensemble AUPRC", 0.005, "p = 0.031, unadjusted", "#bdbdbd")]
for i, (l, d, p, c) in enumerate(rows[::-1]):
    ax.barh(i, d, color=c, height=0.6)
    ax.text(d + 0.0003, i, f"+{d:.3f}  ({p})", va="center", fontsize=7)
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=7)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(0, 0.018)
ax.set_xlabel("ΔAUPRC vs static CM-KELM (paired Wilcoxon over seeds)")
fig.tight_layout()
fig.savefig("f6_multimeme.pdf")

# ---------- Fig. 6: timing (single column, two rows) ----------
fig, (a, b) = plt.subplots(2, 1, figsize=(COLW, 4.2), gridspec_kw=dict(height_ratios=[1.1, 1]))
cpu = [(96, 5.15), (300, 16.89), (1200, 62.16)]; gpu = [(300, 2.95), (750, 7.19)]
for d, c, l, m in [(cpu, "#546e7a", "CPU (24-core EPYC 9224, FP64)", "o"), (gpu, HL, "GPU (RTX 6000 Ada, FP64)", "s")]:
    x, y = zip(*d)
    a.plot(x, y, m + "-", color=c, label=l, ms=3.5)
    for xi, yi in d:
        a.annotate(f"{yi:.2f}" if c == HL else f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(5, -9), fontsize=6.5, color=c)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("population × generations", fontsize=7)
a.set_ylabel("min per search (median)", fontsize=7); a.set_title("(a) Search time vs budget")
a.annotate("×5.7", xy=(300, 3.05), xytext=(255, 9.5), ha="right", va="center", fontsize=7,
           arrowprops=dict(arrowstyle="->", lw=0.8))
a.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100])); a.yaxis.set_major_formatter(ScalarFormatter())
a.yaxis.set_minor_formatter(NullFormatter()); a.set_ylim(2, 100)
a.legend(frameon=False, loc="upper left", fontsize=6.3); a.grid(alpha=0.25, which="both")
labs = ["Preop.\n37 var.", "End of\nsurgery\n96 var.", "Final\n93 var.", "+ tabu", "+ meme\ncontrol.", "+ SA"]
vals = [2.87, 2.94, 2.95, 2.95, 2.94, 2.94]; ns = [10, 50, 10, 50, 40, 10]
b.bar(range(len(vals)), vals, color=[HL] * 3 + ["#f48fb1"] * 3, width=0.65)
for i, (v, n) in enumerate(zip(vals, ns)):
    b.text(i, v + 0.06, f"{v:.2f}\n(n={n})", ha="center", fontsize=6)
b.set_xticks(range(len(vals)), labs, fontsize=6.3); b.set_ylim(0, 4.0); b.set_ylabel("min per search (median)", fontsize=7)
b.set_title("(b) GPU time per search (population 20, 15 generations)", fontsize=7.5)
fig.tight_layout(h_pad=1.0)
fig.savefig("fig_timing.pdf")
print("ok")
