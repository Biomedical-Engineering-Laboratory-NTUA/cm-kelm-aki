"""Forest plot (docs/REPRODUCE.md): paired differences of M1 from every model in the confirmatory cross-validation.
    python fig_forest.py cv_primary.csv f2_cv_forest
"""
import sys, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
def nb(x, ratio=0.25):
    x = np.asarray(x); J = len(x); se = np.sqrt(x.var(ddof=1)*(1/J+ratio)); t = stats.t.ppf(.975, J-1)
    m = x.mean(); return m, m-t*se, m+t*se, 2*stats.t.sf(abs(m/se), J-1)
def holm(p):
    p=np.asarray(p); o=np.argsort(p); a=np.empty_like(p); run=0
    for i,j in enumerate(o): run=max(run,(len(p)-i)*p[j]); a[j]=min(1,run)
    return a
def diff(a,b,m):
    A=d[d.model==a].set_index(["repeat","fold"])[m]; B=d[d.model==b].set_index(["repeat","fold"])[m]
    i=A.index.intersection(B.index); return (A[i]-B[i]).values
lab={"M2":"DE-KELM (mask)","M3":"CM-KELM","M4":"Grid-tuned KELM","M5":"Sparse GP (ARD)","M6":"XGBoost","M7":"FT-Transformer","M8a":"Logistic regression","M8b":"Random forest","LGBM":"LightGBM$^{a}$","RS-ARD":"Random search, ARD$^{b}$","RS-mask":"DE-KELM $-$ random search, mask$^{c}$"}
fam=["M2","M3","M4","M5","M6","M7","M8a","M8b"]
rows=[]
for m in ["auroc","auprc"]:
    R={k:nb(diff("M1",k,m)) for k in fam}
    H=holm([R[k][3] for k in fam])
    for k,h in zip(fam,H): rows.append((k,m)+R[k]+(h,))
    rows.append(("LGBM",m)+nb(diff("M1","LGBM",m))+(np.nan,))
    rows.append(("RS-ARD",m)+nb(diff("M1","RS-ARD",m))+(np.nan,))
    rows.append(("RS-mask",m)+nb(diff("M2","RS-mask",m))+(np.nan,))
T=pd.DataFrame(rows,columns=["model","metric","diff","lo","hi","p","holm"])
T.to_csv(out+"_table.csv",index=False); print(T.round(4).to_string())
order=["M4","M8a","M2","M3","M8b","M5","M7","M6","LGBM","RS-ARD","RS-mask"]
fig,axs=plt.subplots(1,2,figsize=(7.16,2.9),sharey=True)
for ax,m,tt in zip(axs,["auroc","auprc"],["$\\Delta$AUROC, DE-KELM-ARD minus comparator","$\\Delta$AUPRC, DE-KELM-ARD minus comparator"]):
    for i,k in enumerate(order):
        r=T[(T.model==k)&(T.metric==m)].iloc[0]; y=len(order)-1-i
        col="#c0392b" if (not np.isnan(r.holm) and r.holm<0.05) else "0.25"
        if np.isnan(r.holm): col="0.55"
        ax.plot([r.lo,r.hi],[y,y],color=col,lw=1.3); ax.plot(r["diff"],y,"s",color=col,ms=4)
        pv=r.p if np.isnan(r.holm) else r.holm
        ptxt="p<0.001" if pv<0.001 else (f"p={pv:.3f}" if pv<0.01 else f"p={pv:.2f}")
        ax.text(1.02,y,ptxt,transform=ax.get_yaxis_transform(),fontsize=6,va="center")
    ax.axvline(0,color="k",lw=0.7); ax.set_title(tt,fontsize=7.5); ax.tick_params(labelsize=6.5)
    ax.axhline(2.5,color="0.8",lw=0.6)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([lab[k] for k in order][::-1],fontsize=6.8)
fig.tight_layout(w_pad=4.5)
fig.savefig(out+".pdf",bbox_inches="tight"); fig.savefig(out+".png",dpi=200,bbox_inches="tight")
