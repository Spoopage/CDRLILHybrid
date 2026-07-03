"""
analisis_fp_candidates.py
Mengumpulkan dan memprioritaskan kandidat FP dari ketiga agen untuk verifikasi manual.
Output:
  - fp_hotspots_baru.csv      : kandidat baru di luar area GT yang sudah ada
  - fp_hotspots_lengkap.csv   : semua hotspot termasuk yang dekat GT (untuk referensi)
  - fp_peta_kandidat.png      : visualisasi spasial semua FP vs GT
"""
import sys, io, os, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

# ── KONFIGURASI ────────────────────────────────────────────────────────────────
# Set env var SKRIPSI_DATA_ROOT untuk override lokasi folder data (mis. di komputer lain)
SKRIPSI_ROOT   = Path(os.environ.get("SKRIPSI_DATA_ROOT", r"C:\Kuliah\Skripsi\SKRIPSI"))
BASE_INFERENCE = SKRIPSI_ROOT / "Data Pengujian 100 eps"
OUT            = SKRIPSI_ROOT / "Hasil Pengujian" / "output_analisis_revisi"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD_GT   = 2.0   # radius TP (sama dengan analisis utama)
CLUSTER_RADIUS = 5.0   # radius penggabungan FP ke dalam satu hotspot
MIN_REPORTS    = 1     # minimum laporan untuk masuk kandidat (turunkan ke 1 agar semua masuk)

# GT lengkap 12 titik
GT = [
    {"id":"G1","type":"GeometryBug",   "x":-23.4,  "z":-46.58},
    {"id":"G2","type":"GeometryBug",   "x":-33.0,  "z":  5.0 },
    {"id":"G3","type":"GeometryBug",   "x":-18.9,  "z":  7.1 },
    {"id":"G4","type":"GeometryBug",   "x":-32.0,  "z":-44.8 },
    {"id":"N1","type":"NavigationBug", "x": 12.13, "z":  6.43},
    {"id":"N2","type":"NavigationBug", "x":  6.35, "z": -5.11},
    {"id":"N3","type":"NavigationBug", "x":  9.91, "z":  3.92},
    {"id":"N4","type":"NavigationBug", "x":  3.44, "z":-24.43},
    {"id":"N5","type":"NavigationBug", "x":  9.5,  "z":  3.8 },
    {"id":"N6","type":"NavigationBug", "x": 32.0,  "z": 17.0 },
    {"id":"P1","type":"PhysicsBug",    "x":-11.0,  "z":-24.0 },
    {"id":"P2","type":"PhysicsBug",    "x": 13.0,  "z":-35.0 },
]

# Data Map B per agen: CDRL pakai Retest3, IL/Hybrid pakai original
AGENT_FILES = {
    "CDRL":   BASE_INFERENCE / "CDRL"   / "Retest3" / "BugReport_CDRL_Retest3.csv",
    "IL":     BASE_INFERENCE / "IL"     / "BugReport_IL_MapB.csv",
    "Hybrid": BASE_INFERENCE / "Hybrid" / "BugReport_Hybrid_MapB.csv",
}

def euc(x1, z1, x2, z2):
    return math.sqrt((x1-x2)**2 + (z1-z2)**2)

def nearest_gt(x, z):
    dists = [(g["id"], g["type"], euc(x, z, g["x"], g["z"])) for g in GT]
    return min(dists, key=lambda d: d[2])

# ── LOAD DAN GABUNGKAN SEMUA BUG REPORTS ──────────────────────────────────────
print("Memuat data bug report...")
all_fps = []
for agent, path in AGENT_FILES.items():
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        x, z = row["X"], row["Z"]
        gt_id, gt_type, gt_dist = nearest_gt(x, z)
        is_tp = gt_dist <= THRESHOLD_GT
        if not is_tp:
            all_fps.append({
                "agent": agent,
                "type":  row["Type"],
                "X":     x,
                "Z":     z,
                "nearest_gt":   gt_id,
                "dist_to_gt":   round(gt_dist, 2),
            })

fp_df = pd.DataFrame(all_fps)
print(f"  Total FP dari 3 agen: {len(fp_df)} laporan")
print(f"  Breakdown agen: CDRL={len(fp_df[fp_df.agent=='CDRL'])}  "
      f"IL={len(fp_df[fp_df.agent=='IL'])}  "
      f"Hybrid={len(fp_df[fp_df.agent=='Hybrid'])}")

# ── SPATIAL CLUSTERING ────────────────────────────────────────────────────────
# Kelompokkan FP yang berdekatan (dalam CLUSTER_RADIUS) menjadi satu hotspot
print(f"\nMembangun hotspot (cluster radius = {CLUSTER_RADIUS} unit)...")

points = fp_df[["X","Z"]].values
n = len(points)
cluster_id = [-1] * n
current_id = 0

for i in range(n):
    if cluster_id[i] == -1:
        cluster_id[i] = current_id
        for j in range(i+1, n):
            if cluster_id[j] == -1:
                if euc(points[i][0], points[i][1], points[j][0], points[j][1]) <= CLUSTER_RADIUS:
                    cluster_id[j] = current_id
        current_id += 1

fp_df["cluster"] = cluster_id

# Ringkasan per hotspot
hotspots = []
for cid, group in fp_df.groupby("cluster"):
    agents_in = sorted(group["agent"].unique())
    types_in  = group["type"].value_counts().to_dict()
    cx = group["X"].mean()
    cz = group["Z"].mean()
    gt_id, gt_type, gt_dist = nearest_gt(cx, cz)

    # Jarak minimum ke GT mana saja
    min_dist_any_gt = min(euc(cx, cz, g["x"], g["z"]) for g in GT)

    hotspots.append({
        "hotspot_id":     cid,
        "X_pusat":        round(cx, 1),
        "Z_pusat":        round(cz, 1),
        "total_laporan":  len(group),
        "jumlah_agen":    len(agents_in),
        "agen":           ", ".join(agents_in),
        "tipe_bug":       "; ".join(f"{t}:{n}" for t,n in types_in.items()),
        "gt_terdekat":    gt_id,
        "jarak_ke_gt":    round(min_dist_any_gt, 2),
        "status":         "DEKAT GT (≤5u)" if min_dist_any_gt <= 5.0 else "KANDIDAT BARU",
    })

hs_df = pd.DataFrame(hotspots).sort_values(
    ["jumlah_agen","total_laporan"], ascending=False
).reset_index(drop=True)

hs_df.index += 1  # mulai dari 1
hs_new  = hs_df[hs_df["status"] == "KANDIDAT BARU"]
hs_near = hs_df[hs_df["status"] == "DEKAT GT (≤5u)"]

print(f"  Total hotspot     : {len(hs_df)}")
print(f"  Kandidat baru     : {len(hs_new)}  (jarak ke GT > 5 unit)")
print(f"  Dekat GT (≤5u)    : {len(hs_near)}")

# ── PRINT KANDIDAT BARU ───────────────────────────────────────────────────────
print()
print("=" * 72)
print("  KANDIDAT BARU — perlu verifikasi manual di Unity Editor")
print("  (jarak ke GT terdekat > 5 unit, belum terdaftar sebagai GT)")
print("=" * 72)
if len(hs_new) == 0:
    print("  Tidak ada kandidat baru yang memenuhi syarat.")
else:
    for _, row in hs_new.iterrows():
        stars = "★" * row["jumlah_agen"]  # lebih banyak agen = lebih prioritas
        print(f"\n  [{stars}] Hotspot #{row.name} — ({row['X_pusat']:.1f}, {row['Z_pusat']:.1f})")
        print(f"    Laporan  : {row['total_laporan']}x  |  Agen: {row['agen']}")
        print(f"    Tipe bug : {row['tipe_bug']}")
        print(f"    GT terdekat: {row['gt_terdekat']} (jarak {row['jarak_ke_gt']:.2f} unit)")

# ── PRINT DEKAT GT ────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("  DEKAT GT YANG SUDAH ADA (≤5 unit) — kemungkinan area terdampak GT")
print("=" * 72)
for _, row in hs_near.iterrows():
    print(f"  Hotspot ({row['X_pusat']:.1f},{row['Z_pusat']:.1f}): "
          f"{row['total_laporan']}x laporan, agen [{row['agen']}], "
          f"dekat {row['gt_terdekat']} ({row['jarak_ke_gt']:.2f}u)")

# ── SIMPAN CSV ────────────────────────────────────────────────────────────────
hs_new_path  = OUT / "fp_hotspots_baru.csv"
hs_full_path = OUT / "fp_hotspots_lengkap.csv"
hs_new.to_csv(hs_new_path, index=True)
hs_df.to_csv(hs_full_path, index=True)
print(f"\n  [CSV] {hs_new_path}")
print(f"  [CSV] {hs_full_path}")

# ── VISUALISASI ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor("#F5F5F5")

# Plot semua FP sebagai scatter (warna per agen)
agent_colors_fp = {"CDRL": "#F4A460", "IL": "#87CEEB", "Hybrid": "#90EE90"}
for agent, grp in fp_df.groupby("agent"):
    ax.scatter(grp["X"], grp["Z"], s=10, alpha=0.3,
               color=agent_colors_fp[agent], label=f"FP {agent}")

# Tandai pusat hotspot kandidat baru
for _, row in hs_new.iterrows():
    size = 60 + row["total_laporan"] * 20
    ax.scatter(row["X_pusat"], row["Z_pusat"], s=size,
               color="red", marker="X", zorder=6,
               edgecolors="darkred", linewidths=1)
    ax.annotate(f"#{row.name}\n({row['X_pusat']:.0f},{row['Z_pusat']:.0f})",
                (row["X_pusat"], row["Z_pusat"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, color="darkred", fontweight="bold")

# Tandai GT
gt_markers = {"GeometryBug":"^","NavigationBug":"s","PhysicsBug":"D"}
gt_colors  = {"GeometryBug":"#1A6B3C","NavigationBug":"#185FA5","PhysicsBug":"#7F3FBF"}
done = set()
for g in GT:
    t = g["type"]
    lbl = f"GT {t.replace('Bug',' Bug')}" if t not in done else "_"
    ax.scatter(g["x"], g["z"], marker=gt_markers[t], c=gt_colors[t],
               s=120, edgecolors="black", linewidths=1, zorder=7, label=lbl)
    ax.annotate(g["id"], (g["x"], g["z"]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=8, color=gt_colors[t], fontweight="bold")
    done.add(t)

# Lingkaran CLUSTER_RADIUS di sekitar setiap GT (area yang "sudah diketahui")
for g in GT:
    circle = plt.Circle((g["x"], g["z"]), 5.0, fill=False,
                         linestyle="--", color="gray", alpha=0.4, linewidth=0.8)
    ax.add_patch(circle)

ax.set_xlabel("X (unit)"); ax.set_ylabel("Z (unit)")
ax.set_title(f"Peta Kandidat FP untuk Verifikasi Manual\n"
             f"Tanda X merah = kandidat baru (jarak ke GT > 5u)  |  "
             f"Lingkaran abu = radius 5u dari GT", fontsize=11)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

out_img = OUT / "fp_peta_kandidat.png"
plt.tight_layout()
plt.savefig(out_img, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [Chart] {out_img}")
print(f"\n  Selesai. Output di: {OUT}")
