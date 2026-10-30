"""Interpretability heat map (docs/REPRODUCE.md): rank of the 15 variables with the best mean rank under the nine attribution methods.
    python fig_interp.py cv_interp_aki.csv f4_interp
"""
import sys, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
d=pd.read_csv(sys.argv[1]).set_index("variable"); out=sys.argv[2]
cols=list(d.columns)
short=["ARD weights, M1","ARD, GP","Perm., M1","Perm., XGB","Perm., RF","SHAP, M1","SHAP, XGB","SHAP, RF","SHAP, GP"]
order=[0,1,2,5,8,3,6,4,7]   # kernel models first
R=pd.DataFrame({c:stats.rankdata(-d[c].values) for c in cols},index=d.index)
top=R.mean(axis=1).sort_values().index[:15]
nm={"department_UR":"Urological surgery","preop_creatinine":"Creatinine","andur":"Anesthesia duration","uo_ml_kg_h":"Urine output, mL/kg/h","department_NS":"Neurosurgery","cvp_line":"Central venous line","department_GS":"General surgery","preop_albumin":"Albumin","asa":"ASA class","art_line":"Arterial line","sex_M":"Male sex","age":"Age","preop_platelet":"Platelets","department_OS":"Orthopedics","co_monitor":"Cardiac-output monitor","colloid_sum":"Colloids","preop_hb":"Hemoglobin","preop_bun":"Urea nitrogen","map_auc_lt65":"MAP<65 mmHg, area","uo_sum":"Urine output, total","crystalloid_sum":"Crystalloids"}
M=R.loc[top,[cols[i] for i in order]].values
fig,ax=plt.subplots(figsize=(3.5,3.6))
im=ax.imshow(M,cmap="viridis_r",vmin=1,vmax=40,aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j,i,int(M[i,j]),ha="center",va="center",fontsize=5.5,color="black" if M[i,j]<20 else "white")
ax.set_yticks(range(len(top))); ax.set_yticklabels([nm.get(v,v) for v in top],fontsize=6.3)
ax.set_xticks(range(9)); ax.set_xticklabels([short[i] for i in order],rotation=60,ha="right",fontsize=6)
ax.axvline(4.5,color="w",lw=1.5)
cb=fig.colorbar(im,ax=ax,shrink=0.55,pad=0.02,extend="max",ticks=[1,10,20,30,40]); cb.ax.set_yticklabels(["1","10","20","30","\u226540"]); cb.ax.tick_params(labelsize=5.5); cb.set_label("rank (1 = most important)",fontsize=6)
fig.tight_layout(); fig.savefig(out+".pdf",bbox_inches="tight"); fig.savefig(out+".png",dpi=220,bbox_inches="tight")
print(list(top))
