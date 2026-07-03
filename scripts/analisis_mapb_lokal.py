"""
analisis_mapb.py 
========================
- TIMESCALE = 1
- Generalization Gap: rata-rata coverage per episode Map A vs rata-rata per episode Map B
- Total grid Map A = 4640
- AUC 0-500k steps dengan trapezoid rule
- Semua chart untuk Bab 4 + chart Map A training
"""

import sys, io, os, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================

# Set env var SKRIPSI_DATA_ROOT untuk override lokasi folder data (mis. di komputer lain)
SKRIPSI_ROOT      = Path(os.environ.get("SKRIPSI_DATA_ROOT", r"C:\Kuliah\Skripsi\SKRIPSI"))
BASE_INFERENCE    = SKRIPSI_ROOT / "Data Pengujian 100 eps"
BASE_TRAINING     = SKRIPSI_ROOT / "Data Training"
TOTAL_MAP_A       = 4640
TOTAL_MAP_B       = 1161
TIMESCALE         = 5

# Map A TotalCoverage:
#   CDRL  → Retraining_04 (reward diperbaiki 0.1/0.3, dipakai untuk inference Retest3)
#   IL    → Training_03 (tidak berubah, extrinsic reward = 0)
#   Hybrid→ Training_01 (reward sudah benar 0.1/0.3 sejak awal)
MAPA_FILES = {
    "CDRL":   BASE_TRAINING / "CDRL" / "Retraining 4" / "CDRL_Retraining_04_CDRLAgent_TotalCoverage.csv",
    "IL":     BASE_TRAINING / "IL" / "IL_Training_03_ILAgent_TotalCoverage.csv",
    "Hybrid": BASE_TRAINING / "Hybrid" / "Hybrid_Training_01_HybridAgent_TotalCoverage.csv",
}

GROUND_TRUTH = [
    {"id":"G1","type":"GeometryBug",   "x":-23.4, "z":-46.58},
    {"id":"G2","type":"GeometryBug",   "x":-33.0, "z":  5.0 },
    {"id":"N1","type":"NavigationBug", "x": 12.13,"z":  6.43},
    {"id":"N2","type":"NavigationBug", "x":  6.35,"z": -5.11},
    {"id":"N3","type":"NavigationBug", "x":  9.91,"z":  3.92},
    {"id":"N4","type":"NavigationBug", "x":  3.44,"z":-24.43},
    {"id":"P1","type":"PhysicsBug",    "x":-11.0, "z":-24.0 },
    {"id":"P2","type":"PhysicsBug",    "x": 13.0, "z":-35.0 },
]
EXTRA_GT    = [
    {"id":"G3","type":"GeometryBug",   "x":-18.9, "z":  7.1 },
    {"id":"G4","type":"GeometryBug",   "x":-32.0, "z":-44.8 },
    {"id":"N5","type":"NavigationBug", "x":  9.5, "z":  3.8 },
    {"id":"N6","type":"NavigationBug", "x": 32.0, "z": 17.0 },
]
ALL_GT      = GROUND_TRUTH + EXTRA_GT
AGENTS      = ["CDRL", "IL", "Hybrid"]
BUG_TYPES   = ["GeometryBug", "NavigationBug", "PhysicsBug"]
THRESHOLD   = 2.0
AGENT_COLORS = {"CDRL":"#D85A30","IL":"#185FA5","Hybrid":"#0F6E56"}
OUTPUT_DIR  = SKRIPSI_ROOT / "Hasil Pengujian" / "output_analisis_redemo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def euc(x1,z1,x2,z2): return math.sqrt((x1-x2)**2+(z1-z2)**2)
def trap(y,x): return float(np.trapz(y,x))

def load_mapb(agent):
    # CDRL: pakai Retest3 (reward diperbaiki), IL/Hybrid: data original
    if agent == "CDRL":
        p = Path(BASE_INFERENCE) / "CDRL" / "Retest3"
        m = "CDRL_Retest3"
    else:
        p = Path(BASE_INFERENCE) / agent
        m = f"{agent}_MapB"
    return {k: pd.read_csv(p/f"{k}_{m}.csv")
            for k in ("InferenceLog","CoverageData","BugReport","PositionLog")}

def detect(bug_df, gt_list, type_filter=None):
    if type_filter:
        bugs = bug_df[bug_df["Type"].str.contains(type_filter,case=False,na=False)]
        gts  = [g for g in gt_list if type_filter in g["type"]]
    else:
        bugs, gts = bug_df, gt_list
    tp_ids, fp_list = set(), []
    for _,row in bugs.iterrows():
        hit = next((g["id"] for g in gts
                    if euc(row["X"],row["Z"],g["x"],g["z"])<=THRESHOLD), None)
        if hit: tp_ids.add(hit)
        else:   fp_list.append(dict(episode=row["Episode"],type=row["Type"],
                                    x=row["X"],z=row["Z"]))
    tp,fn,fp = len(tp_ids),len(gts)-len(tp_ids),len(fp_list)
    p  = tp/(tp+fp) if tp+fp>0 else 0.0
    r  = tp/(tp+fn) if tp+fn>0 else 0.0
    f1 = 2*p*r/(p+r) if p+r>0 else 0.0
    return dict(tp=tp,fp=fp,fn=fn,tp_ids=sorted(tp_ids),
                precision=p,recall=r,f1=f1,fp_details=fp_list)

def saturation(cov_df, final_n):
    cumul, res = set(), {50:None,75:None,90:None}
    for ep in sorted(cov_df["Episode"].unique()):
        cumul |= set(map(tuple,cov_df[cov_df["Episode"]==ep][["X","Z"]].values))
        pct = len(cumul)/final_n*100
        for t in [50,75,90]:
            if res[t] is None and pct>=t: res[t]=ep
    return res

# ============================================================
# CHARTS
# ============================================================

def chart_mapa_training(mapa_data, out):
    """Grafik total coverage Map A per episode ketiga agen."""
    fig,ax = plt.subplots(figsize=(10,5))
    for agent in AGENTS:
        df = mapa_data[agent]['df']
        df_roll = df.copy()
        df_roll['roll10'] = df['Value'].rolling(10,min_periods=3).mean()
        ax.scatter(df['Step']/1e6, df['Value'],
                   alpha=0.25, s=12, color=AGENT_COLORS[agent])
        ax.plot(df_roll['Step']/1e6, df_roll['roll10'],
                color=AGENT_COLORS[agent], linewidth=2, label=agent)
    ax.set_xlabel("Training steps (juta)")
    ax.set_ylabel("Total coverage per episode (grid)")
    ax.set_title("Perkembangan Total Coverage selama Pelatihan di Map A")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines[["top","right"]].set_visible(False)
    path = out/"mapa_training_coverage.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_mapa_auc_comparison(mapa_data, out):
    """Zoom 0-500k untuk visualisasi AUC dan cold start."""
    fig,ax = plt.subplots(figsize=(9,5))
    for agent in AGENTS:
        df = mapa_data[agent]['df']
        early = df[df['Step']<=500000].copy()
        early['roll3'] = early['Value'].rolling(3,min_periods=1).mean()
        ax.scatter(early['Step']/1e3, early['Value'],
                   alpha=0.4, s=20, color=AGENT_COLORS[agent])
        ax.plot(early['Step']/1e3, early['roll3'],
                color=AGENT_COLORS[agent], linewidth=2,
                label=f"{agent}  (AUC={mapa_data[agent]['auc']/1e6:.1f}M)")
        # Fill under curve
        ax.fill_between(early['Step']/1e3, early['Value'],
                        alpha=0.08, color=AGENT_COLORS[agent])
    ax.axvline(500, color='gray', linestyle=':', linewidth=1,
               label="Batas fase BC Hybrid (500k)")
    ax.set_xlabel("Training steps (ribu)")
    ax.set_ylabel("Total coverage per episode (grid)")
    ax.set_title("AUC Coverage 0–500.000 Steps Pertama (Perbandingan Cold Start)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines[["top","right"]].set_visible(False)
    path = out/"mapa_auc_500k.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_gen_gap(summary_list, out):
    """Bar chart generalization gap per agen."""
    agents = [r["Agen"] for r in summary_list]
    gaps   = [r["Gen_Gap_pct"] for r in summary_list]
    colors = [AGENT_COLORS[a] for a in agents]
    bar_colors = [("#0F6E56" if g>=0 else "#993C1D") for g in gaps]

    fig,ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(agents, gaps, color=bar_colors, width=0.5, edgecolor="white")
    ax.axhline(0, color='black', linewidth=0.8)
    for bar,val in zip(bars,gaps):
        ypos = bar.get_height()+0.05 if val>=0 else bar.get_height()-0.25
        ax.text(bar.get_x()+bar.get_width()/2, ypos,
                f"{val:+.2f} poin", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Generalization Gap (poin persentase)")
    ax.set_title("Generalization Gap — Rata-rata Coverage Map A vs Map B\n"
                 "(positif = penurunan ke Map B, negatif = peningkatan)")
    ax.spines[["top","right"]].set_visible(False)
    path = out/"gen_gap_MapB.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_mapa_mapb_comparison(summary_list, out):
    """Grouped bar chart: coverage Map A vs Map B per agen."""
    agents    = [r["Agen"] for r in summary_list]
    mapa_vals = [r["Mean_Coverage_MapA_pct"] for r in summary_list]
    mapb_vals = [r["Mean_Coverage_MapB_pct"] for r in summary_list]
    x     = np.arange(len(agents))
    width = 0.35
    fig,ax = plt.subplots(figsize=(8,5))
    bars_a = ax.bar(x-width/2, mapa_vals, width, label="Map A (training)",
                    color=[AGENT_COLORS[a] for a in agents], alpha=0.6,
                    edgecolor="white", hatch="//")
    bars_b = ax.bar(x+width/2, mapb_vals, width, label="Map B (testing)",
                    color=[AGENT_COLORS[a] for a in agents],
                    edgecolor="white")
    for bar,val in zip(list(bars_a)+list(bars_b), mapa_vals+mapb_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(agents)
    ax.set_ylabel("Rata-rata Coverage per Episode (%)")
    ax.set_title("Perbandingan Rata-rata Coverage per Episode:\nMap A (Pelatihan) vs Map B (Pengujian)")
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    # Custom legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.6, hatch='//', label='Map A (training)'),
        Patch(facecolor='gray', label='Map B (testing)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    path = out/"mapa_vs_mapb_coverage.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_heatmap(pos_df, agent, cov_pct, out):
    fig,ax = plt.subplots(figsize=(7,7))
    h = ax.hist2d(pos_df["X"],pos_df["Z"],bins=80,cmap="YlOrRd",density=False)
    plt.colorbar(h[3],ax=ax,label="Frekuensi kunjungan")
    markers = {"GeometryBug":("^","#1A6B3C","Geometry GT"),
               "NavigationBug":("s","#185FA5","Navigation GT"),
               "PhysicsBug":("D","#7F3FBF","Physics GT")}
    done = set()
    for g in ALL_GT:
        mk,col,lbl = markers[g["type"]]
        ax.scatter(g["x"],g["z"],marker=mk,c=col,s=90,
                   edgecolors="white",linewidths=0.8,zorder=5,
                   label=(lbl if lbl not in done else "_"))
        ax.annotate(g["id"],(g["x"],g["z"]),textcoords="offset points",
                    xytext=(5,5),fontsize=7,color=col,fontweight="bold")
        done.add(lbl)
    ax.set_xlabel("X (unit)"); ax.set_ylabel("Z (unit)")
    ax.set_title(f"Spatial Heatmap Trajektori — Agen {agent} (Map B)\n"
                 f"Coverage union: {cov_pct:.2f}%  |  {len(pos_df):,} posisi direkam",
                 fontsize=11)
    ax.legend(loc="upper right",fontsize=8,framealpha=0.85)
    ax.set_aspect("equal")
    path = out/f"heatmap_{agent}_MapB.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Heatmap] {path}")

def chart_coverage_comparison(summary_list, out):
    agents    = [r["Agen"] for r in summary_list]
    coverages = [r["Coverage_MapB_union_pct"] for r in summary_list]
    fig,ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(agents, coverages,
                  color=[AGENT_COLORS[a] for a in agents],
                  width=0.5, edgecolor="white")
    ax.axhline(40,color="#BA7517",linestyle="--",linewidth=1.5,label="Target RM1 (40%)")
    for bar,val in zip(bars,coverages):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Coverage Absolut — Union 100 Episode (%)")
    ax.set_title("Perbandingan Coverage Absolut di Map B (100 Episode)")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    path = out/"coverage_comparison_MapB.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_cumulative_coverage(all_cov, out):
    fig,ax = plt.subplots(figsize=(9,5))
    for agent in AGENTS:
        cov = all_cov[agent]
        cumul,eps,vals = set(),[],[]
        for ep in sorted(cov["Episode"].unique()):
            cumul |= set(map(tuple,cov[cov["Episode"]==ep][["X","Z"]].values))
            eps.append(ep); vals.append(len(cumul)/TOTAL_MAP_B*100)
        ax.plot(eps,vals,color=AGENT_COLORS[agent],linewidth=2,label=agent)
    ax.axhline(40,color="#BA7517",linestyle="--",linewidth=1.2,label="Target RM1 (40%)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Coverage Kumulatif (%)")
    ax.set_title("Akumulasi Coverage Kumulatif per Episode — Map B (100 Episode)")
    ax.legend(fontsize=10); ax.grid(True,alpha=0.2)
    ax.spines[["top","right"]].set_visible(False)
    path = out/"cumulative_coverage_MapB.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_saturation(summary_list, out):
    labels = ["50%","75%","90%"]
    x,width = np.arange(len(labels)),0.25
    fig,ax = plt.subplots(figsize=(7,4))
    for i,agent in enumerate(AGENTS):
        row  = next(r for r in summary_list if r["Agen"]==agent)
        vals = [row["Sat_50ep"],row["Sat_75ep"],row["Sat_90ep"]]
        bars = ax.bar(x+(i-1)*width,vals,width,label=agent,
                      color=AGENT_COLORS[agent],edgecolor="white")
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                    str(int(v)),ha="center",va="bottom",fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Ep. capai 50%","Ep. capai 75%","Ep. capai 90%"])
    ax.set_ylabel("Nomor Episode")
    ax.set_title("Analisis Saturasi Coverage — Map B")
    ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
    path = out/"saturation_analysis_MapB.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_efficiency(summary_list, out):
    agents = [r["Agen"] for r in summary_list]
    vals   = [r["Efisiensi_grid_per_menit"] for r in summary_list]
    fig,ax = plt.subplots(figsize=(7,4))
    bars   = ax.bar(agents,vals,
                    color=[AGENT_COLORS[a] for a in agents],
                    width=0.5,edgecolor="white")
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                f"{val:.1f}",ha="center",va="bottom",fontsize=11,fontweight="bold")
    ax.set_ylabel(f"Grid unik / menit waktu nyata (timeScale={TIMESCALE})")
    ax.set_title("Efisiensi Coverage per Agen — Map B (100 Episode)")
    ax.spines[["top","right"]].set_visible(False)
    path = out/"efficiency_coverage_MapB.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_recall(all_metrics, out):
    x,width = np.arange(len(BUG_TYPES)),0.25
    fig,ax = plt.subplots(figsize=(9,5))
    for i,agent in enumerate(AGENTS):
        recalls = [all_metrics[agent]["by_type"][bt]["recall"] for bt in BUG_TYPES]
        bars = ax.bar(x+(i-1)*width,recalls,width,label=agent,
                      color=AGENT_COLORS[agent],edgecolor="white")
        for bar,v in zip(bars,recalls):
            if v>0:
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                        f"{v:.2f}",ha="center",va="bottom",fontsize=8)
    ax.axhline(0.6,color="#BA7517",linestyle="--",linewidth=1.5,label="Target RM3 (60%)")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("Bug"," Bug") for b in BUG_TYPES])
    ax.set_ylabel("Recall"); ax.set_ylim(0,1.1)
    ax.set_title("Perbandingan Recall per Tipe Anomali — Map B")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    path = out/"recall_per_bugtype_MapB.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_episode_duration(all_inf, out):
    fig,ax = plt.subplots(figsize=(9,4))
    for agent in AGENTS:
        inf = all_inf[agent]
        ax.plot(inf["Episode"].values,inf["DurasiEpisode_s"].values,
                alpha=0.6,linewidth=0.8,label=agent,color=AGENT_COLORS[agent])
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Durasi game (detik, timeScale={TIMESCALE})")
    ax.set_title(f"Durasi Episode per Agen — Map B (100 Episode, timeScale={TIMESCALE})")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    path = out/"episode_duration_MapB.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [Chart] {path}")

def chart_gt_detection(all_bug, out):
    fig,axes = plt.subplots(1,3,figsize=(15,6))
    for idx,agent in enumerate(AGENTS):
        ax = axes[idx]
        bug_df = all_bug[agent]
        ax.set_facecolor("#F8F8F8")
        ax.scatter(bug_df["X"],bug_df["Z"],c="#CCCCCC",s=15,alpha=0.5,
                   zorder=2,label="Laporan bug (FP)")
        tp_bugs = []
        for _,row in bug_df.iterrows():
            for g in ALL_GT:
                if euc(row["X"],row["Z"],g["x"],g["z"])<=THRESHOLD:
                    tp_bugs.append((row["X"],row["Z"],g["type"])); break
        if tp_bugs:
            tp_df = pd.DataFrame(tp_bugs,columns=["X","Z","type"])
            tc = {"GeometryBug":"#1A6B3C","NavigationBug":"#185FA5","PhysicsBug":"#7F3FBF"}
            for t,col in tc.items():
                sub = tp_df[tp_df["type"]==t]
                if len(sub):
                    ax.scatter(sub["X"],sub["Z"],c=col,s=80,zorder=4,
                               marker="*",label=f"TP {t.replace('Bug','')}")
        gs = {"GeometryBug":"^","NavigationBug":"s","PhysicsBug":"D"}
        gc = {"GeometryBug":"#1A6B3C","NavigationBug":"#185FA5","PhysicsBug":"#7F3FBF"}
        done = set()
        for g in ALL_GT:
            lbl = f"GT {g['type'].replace('Bug','')}"
            ax.scatter(g["x"],g["z"],marker=gs[g["type"]],c=gc[g["type"]],
                       s=120,edgecolors="black",linewidths=1,zorder=5,
                       label=(lbl if lbl not in done else "_"))
            ax.annotate(g["id"],(g["x"],g["z"]),textcoords="offset points",
                        xytext=(4,4),fontsize=7,fontweight="bold",
                        color=gc[g["type"]])
            done.add(lbl)
        ax.set_title(f"Agen {agent}",fontsize=12,fontweight="bold",
                     color=AGENT_COLORS[agent])
        ax.set_xlabel("X (unit)"); ax.set_ylabel("Z (unit)")
        ax.set_xlim(-55,45); ax.set_ylim(-70,25)
        ax.legend(fontsize=7,loc="upper right",framealpha=0.85)
        ax.grid(True,alpha=0.2)
    fig.suptitle("Peta Deteksi Anomali — Ground Truth vs Laporan Bug (Map B)",
                 fontsize=13,fontweight="bold")
    path = out/"gt_detection_map_MapB.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  [GT Map] {path}")

# ============================================================
# MAIN
# ============================================================

print("="*65)
print(f"ANALISIS LENGKAP MAP A + MAP B  (timeScale={TIMESCALE})")
print(f"Map A: {TOTAL_MAP_A} grid  |  Map B: {TOTAL_MAP_B} grid")
print(f"Threshold deteksi: {THRESHOLD} unit  |  GT: {len(ALL_GT)} titik")
print("="*65)

# Load Map A data
print("\nMemuat data TensorBoard Map A...")
mapa_data = {}
for agent in AGENTS:
    path = MAPA_FILES[agent]
    df   = pd.read_csv(path).sort_values('Step').reset_index(drop=True)
    early = df[df['Step']<=500000].copy()
    auc   = trap(early['Value'].values, early['Step'].values)
    mean_a = df['Value'].mean()
    mean_a_pct = mean_a / TOTAL_MAP_A * 100
    ph = {
        '0-500k':    df[df['Step']<=500000]['Value'].mean(),
        '500k-2.5M': df[(df['Step']>500000)&(df['Step']<=2500000)]['Value'].mean(),
        '2.5M-5M':   df[df['Step']>2500000]['Value'].mean(),
    }
    cv_early = early['Value'].std()/early['Value'].mean()*100 if early['Value'].mean()>0 else 0
    mapa_data[agent] = dict(
        df=df, early=early, auc=auc,
        mean_grid=mean_a, mean_pct=mean_a_pct,
        phases=ph, cv_early=cv_early
    )
    print(f"  {agent}: mean={mean_a:.1f}grid={mean_a_pct:.2f}%  "
          f"AUC={auc/1e6:.2f}M  CV_early={cv_early:.1f}%")

# RM4
auc_c = mapa_data['CDRL']['auc']
auc_h = mapa_data['Hybrid']['auc']
rm4_diff = auc_h - auc_c
rm4_pct  = rm4_diff/auc_c*100
print(f"\n  RM4 — AUC Hybrid vs CDRL: {rm4_diff:+,.0f} ({rm4_pct:+.1f}%)")
print(f"  RM4 target ≥+20%: {'✅ PASS' if rm4_pct>=20 else '❌ FAIL (CDRL lebih tinggi karena eksplorasi stokastik awal)'}")

# Load Map B data
print("\nMemuat data inference Map B...")
summary_list = []
all_metrics, all_inf, all_cov, all_bug = {},{},{},{}
fp_rows = []

for agent in AGENTS:
    data = load_mapb(agent)
    inf,cov,bug,pos = (data["InferenceLog"],data["CoverageData"],
                       data["BugReport"],data["PositionLog"])
    all_inf[agent]=inf; all_cov[agent]=cov; all_bug[agent]=bug

    # Coverage union
    ug       = cov[["X","Z"]].drop_duplicates()
    n_union  = len(ug)
    pct_union = n_union/TOTAL_MAP_B*100

    # Coverage mean per episode Map B
    per_ep_b  = cov.groupby('Episode').apply(
        lambda g: len(g[['X','Z']].drop_duplicates()), include_groups=False)
    mean_b    = per_ep_b.mean()
    mean_b_pct = mean_b/TOTAL_MAP_B*100

    # Generalization gap (mean vs mean)
    gen_gap = mapa_data[agent]['mean_pct'] - mean_b_pct

    # Saturation
    sat = saturation(cov, n_union)

    # Detection
    agg      = detect(bug, ALL_GT)
    by_type  = {bt: detect(bug,ALL_GT,type_filter=bt) for bt in BUG_TYPES}
    all_metrics[agent] = {"agg":agg,"by_type":by_type}

    # Efficiency
    total_real_s  = inf['WaktuKumulatif_s'].max()/TIMESCALE
    efficiency    = n_union/(total_real_s/60)
    ep_mean_real  = inf['DurasiEpisode_s'].mean()/TIMESCALE
    ep_max_real   = inf['DurasiEpisode_s'].max()/TIMESCALE
    missing_cov   = set(range(1,101))-set(cov['Episode'].unique())

    if agg['fp_details']:
        fp_rows.append(pd.DataFrame(agg['fp_details']).assign(agent=agent))

    print(f"\n  [{agent}]")
    print(f"    Coverage union     : {n_union} grid = {pct_union:.2f}%  "
          f"{'✅' if pct_union>=40 else '❌'} RM1")
    print(f"    Coverage mean/ep   : {mean_b:.1f} grid = {mean_b_pct:.2f}%")
    print(f"    Generalization Gap : {gen_gap:+.2f} poin % "
          f"(MapA {mapa_data[agent]['mean_pct']:.2f}% → MapB {mean_b_pct:.2f}%)")
    print(f"    Deteksi GT         : {agg['tp_ids']} TP={agg['tp']}/{len(ALL_GT)}  "
          f"R={agg['recall']:.3f}  {'✅' if agg['recall']>=0.6 else '❌'} RM3")
    print(f"    Efisiensi          : {efficiency:.1f} grid/menit nyata")
    print(f"    Saturasi           : 50%→ep{sat[50]}  75%→ep{sat[75]}  90%→ep{sat[90]}")
    print(f"    Durasi mean/max    : {ep_mean_real:.2f}s/{ep_max_real:.2f}s nyata")

    chart_heatmap(pos, agent, pct_union, OUTPUT_DIR)

    summary_list.append({
        "Agen": agent,
        "Mean_Coverage_MapA_grid": round(mapa_data[agent]['mean_grid'],1),
        "Mean_Coverage_MapA_pct":  round(mapa_data[agent]['mean_pct'],2),
        "Coverage_MapB_union_grid": n_union,
        "Coverage_MapB_union_pct": round(pct_union,2),
        "Mean_Coverage_MapB_grid": round(mean_b,1),
        "Mean_Coverage_MapB_pct":  round(mean_b_pct,2),
        "Gen_Gap_pct":             round(gen_gap,2),
        "AUC_0_500k":              round(mapa_data[agent]['auc']/1e6,2),
        "CV_early_0_500k_pct":     round(mapa_data[agent]['cv_early'],1),
        "Ep_Missing_Coverage":     len(missing_cov),
        "Ep_Durasi_Mean_Real_s":   round(ep_mean_real,2),
        "Ep_Durasi_Max_Real_s":    round(ep_max_real,2),
        "Efisiensi_grid_per_menit": round(efficiency,1),
        "Sat_50ep": sat[50], "Sat_75ep": sat[75], "Sat_90ep": sat[90],
        "TP_Total": agg['tp'], "FP_Total": agg['fp'], "FN_Total": agg['fn'],
        "GT_Detected": ", ".join(agg['tp_ids']) if agg['tp_ids'] else "-",
        "Precision": round(agg['precision'],3),
        "Recall":    round(agg['recall'],3),
        "F1":        round(agg['f1'],3),
        "Recall_Geometry":   round(by_type['GeometryBug']['recall'],3),
        "Recall_Navigation": round(by_type['NavigationBug']['recall'],3),
        "Recall_Physics":    round(by_type['PhysicsBug']['recall'],3),
    })

# Charts
print("\nMembuat semua chart...")
chart_mapa_training(mapa_data, OUTPUT_DIR)
chart_mapa_auc_comparison(mapa_data, OUTPUT_DIR)
chart_coverage_comparison(summary_list, OUTPUT_DIR)
chart_cumulative_coverage(all_cov, OUTPUT_DIR)
chart_mapa_mapb_comparison(summary_list, OUTPUT_DIR)
chart_gen_gap(summary_list, OUTPUT_DIR)
chart_saturation(summary_list, OUTPUT_DIR)
chart_efficiency(summary_list, OUTPUT_DIR)
chart_recall(all_metrics, OUTPUT_DIR)
chart_episode_duration(all_inf, OUTPUT_DIR)
chart_gt_detection(all_bug, OUTPUT_DIR)

# CSVs
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(OUTPUT_DIR/"ringkasan_metrik_final.csv", index=False)
print(f"  [CSV] ringkasan_metrik_final.csv")

if fp_rows:
    fp_all = pd.concat(fp_rows,ignore_index=True)
    fp_all['xr']=fp_all['x'].round(0); fp_all['zr']=fp_all['z'].round(0)
    cluster = (fp_all.groupby(['type','xr','zr'])
               .agg(total=('agent','count'),
                    agen=('agent',lambda s:", ".join(sorted(set(s)))))
               .reset_index().sort_values('total',ascending=False))
    cluster.to_csv(OUTPUT_DIR/"fp_candidates_for_verification.csv", index=False)
    print(f"  [CSV] fp_candidates_for_verification.csv")

# Print final summary
print("\n"+"="*65)
print("RINGKASAN FINAL")
print("="*65)
print(summary_df[[
    "Agen","Mean_Coverage_MapA_pct","Mean_Coverage_MapB_pct","Gen_Gap_pct",
    "Coverage_MapB_union_pct","AUC_0_500k","Recall","Efisiensi_grid_per_menit"
]].to_string(index=False))

print(f"\nRM4 — AUC Hybrid vs CDRL: {rm4_pct:+.1f}%  "
      f"({'✅ PASS' if rm4_pct>=20 else '❌ FAIL'})")
print(f"\n✅ Selesai. {len(list(OUTPUT_DIR.glob('*.png')))} chart + "
      f"{len(list(OUTPUT_DIR.glob('*.csv')))} CSV tersimpan di: {OUTPUT_DIR}")
