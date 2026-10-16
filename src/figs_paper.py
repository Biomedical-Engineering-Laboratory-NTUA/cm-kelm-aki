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
M = [("Logistic regression", .8605, .8475, .8737, .2680, .2421, .2990),
     ("TabTransformer", .8679, .8549, .8808, .3052, .2768, .3384),
     ("Grid-tuned KELM", .8750, .8626, .8870, .2926, .2660, .3225),
     ("Tabular Mamba", .8756, .8636, .8878, .2817, .2563, .3124),
     ("Random forest", .8821, .8703, .8938, .3389, .3093, .3720),
     ("Stacked DL", .8832, .8716, .8949, .3122, .2840, .3453),
     ("CM-KELM", .8866, .8746, .8977, .3403, .3112, .3740),
     ("JIT Transformer", .8899, .8788, .9006, .3366, .3072, .3690),
     ("LightGBM", .8930, .8811, .9042, .3879, .3559, .4237),
     ("XGBoost", .8929, .8808, .9037, .4071, .3743, .4403)]
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
D = [("Logistic regression", -.0144, -.0216, -.0073, 4.06e-4, -.0247, -.0501, .0004, 0.212),
     ("TabTransformer", -.0070, -.0139, -.0002, 0.108, .0125, -.0114, .0374, 0.61),
     ("Tabular Mamba", .0006, -.0063, .0075, 0.862, -.0109, -.0354, .0132, 0.61),
     ("Random forest", .0071, .0005, .0137, 0.108, .0463, .0191, .0732, 0.0045),
     ("Stacked DL", .0082, .0027, .0137, 0.0133, .0195, -.0031, .0433, 0.255),
     ("CM-KELM", .0116, .0074, .0159, 8.01e-7, .0477, .0276, .0682, 0.0045),
     ("JIT Transformer", .0149, .0091, .0207, 2.41e-6, .0440, .0188, .0692, 0.010),
     ("LightGBM", .0180, .0114, .0246, 8.01e-7, .0953, .0682, .1224, 0.0045),
     ("XGBoost", .0179, .0115, .0242, 3.04e-7, .1144, .0875, .1397, 0.0045)]
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
data = {"AUROC": {"Grid-tuned KELM": (.853, .875), "CM-KELM": (.860, .887), "XGBoost": (.876, .893)},
        "AUPRC": {"Grid-tuned KELM": (.265, .293), "CM-KELM": (.286, .340), "XGBoost": (.340, .407)}}
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
     ("Arterial line", 9, 1), ("Cardiac-output monitoring", 9, 1), ("Urine output, mL/kg/h", 9, 1),
     ("ASA class", 8, 0), ("Insulin (preop.)", 8, 0), ("Urine output, total", 8, 1),
     ("Time with temperature <36 °C", 7, 1), ("Orthopedic surgery", 7, 0), ("Otorhinolaryngology", 7, 0)]
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
        a.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(5, -9), fontsize=6.5, color=c)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("population × generations", fontsize=7)
a.set_ylabel("min per search (median)", fontsize=7); a.set_title("(a) Search time vs budget")
a.annotate("×5.6", xy=(300, 3.05), xytext=(255, 9.5), ha="right", va="center", fontsize=7,
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
