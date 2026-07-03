"""
checkpoint_selector.py
======================
Analisis pemilihan checkpoint terbaik berdasarkan metrik TensorBoard.

Logika pemilihan per tipe agen:
  - RL / CDRL : TotalCoverage (50%) + ExplorationRatio (30%) + CumulativeReward (20%)
                Semua metrik: lebih tinggi = lebih baik.
  - IL (BC)   : PolicyLoss sebagai metrik utama (lebih RENDAH = lebih baik).
                TotalCoverage ditampilkan sebagai referensi saja — pada IL
                nilainya stabil dan tidak membedakan antar checkpoint.
                Catatan: IL rentan overfitting/overspecialization terhadap
                jalur demo; early stopping direkomendasikan jika PolicyLoss
                mulai naik kembali.

Cara pakai:
    python checkpoint_selector.py

Sesuaikan variabel CONFIGS di bawah untuk run yang berbeda.
"""

import sys
import io
import csv
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Set env var SKRIPSI_DATA_ROOT untuk override lokasi folder data (mis. di komputer lain)
SKRIPSI_ROOT  = os.environ.get("SKRIPSI_DATA_ROOT", r"C:\Kuliah\Skripsi\SKRIPSI")
TRAINING_ROOT = os.path.join(SKRIPSI_ROOT, "Data Training")

CONFIGS = [
    # ── MODEL LAMA (Training awal, dipakai untuk MapB inference pertama) ──
    {
        "label":        "CDRL_Training_08",
        "agent_type":   "RL",
        "note":         "Model lama — dipakai MapB inference Juni 6. Tidak ada ExplorationRatio.",
        "base_dir":     os.path.join(TRAINING_ROOT, "CDRL"),
        "prefix":       "CDRL_Training_08_CDRLAgent",
        # Checkpoint .onnx yang tersimpan di results/CDRL_Training_08/CDRLAgent/
        # CDRLAgent.onnx (final, dipakai inference) = salinan dari 5000037
        "checkpoints":  [3_499_918, 3_999_943, 4_499_977, 4_999_909, 5_000_037],
        "ckpt_rewards": {3_499_918: None, 3_999_943: None, 4_499_977: None,
                         4_999_909: None, 5_000_037: None},
        "final_step":   5_000_037,
        "metric_aliases": {"CumulativeReward": "Cummulative_Reward"},
    },
    {
        "label":        "Hybrid_Training_01",
        "agent_type":   "RL",
        "note":         "Model lama — dipakai MapB inference Juni 6. Tidak ada ExplorationRatio.",
        "base_dir":     os.path.join(TRAINING_ROOT, "Hybrid"),
        "prefix":       "Hybrid_Training_01_HybridAgent",
        # Checkpoint .onnx yang tersimpan di results/Hybrid_Training_01/HybridAgent/
        # HybridAgent.onnx (final, dipakai inference) = salinan dari 5000046
        "checkpoints":  [3_499_944, 3_999_890, 4_499_959, 4_999_918, 5_000_046],
        "ckpt_rewards": {3_499_944: None, 3_999_890: None, 4_499_959: None,
                         4_999_918: None, 5_000_046: None},
        "final_step":   5_000_046,
        "metric_aliases": {"CumulativeReward": "Cummulative_Reward"},
    },
    # ── IL ────────────────────────────────────────────────────────────────
    {
        "label":        "IL_Training_03",
        "agent_type":   "IL",           # pakai PolicyLoss sebagai kriteria
        "base_dir":     os.path.join(TRAINING_ROOT, "IL"),
        "prefix":       "IL_Training_03_ILAgent",
        "checkpoints":  [3_999_967, 4_499_974, 4_663_529, 4_999_958, 5_000_022],
        "ckpt_rewards": {               # reward IL selalu 0, tidak dipakai seleksi
            3_999_967: 0.0,
            4_499_974: 0.0,
            4_663_529: 0.0,
            4_999_958: 0.0,
            5_000_022: 0.0,
        },
        "final_step":   5_000_022,
    },
    # ── CDRL ──────────────────────────────────────────────────────────────
    {
        "label":        "CDRL_Retraining_03",
        "agent_type":   "RL",           # pakai TotalCoverage + ExplRatio + Reward
        "base_dir":     os.path.join(TRAINING_ROOT, "CDRL", "Retraining"),
        "prefix":       "CDRL_Retraining_03_CDRLAgent",
        "checkpoints":  [3_499_979, 3_999_915, 4_499_919, 4_999_901, 5_000_029],
        "ckpt_rewards": {
            3_499_979: None,
            3_999_915: 83.2,
            4_499_919: 109.0,
            4_999_901: 103.0,
            5_000_029: 103.0,
        },
        "final_step":   5_000_029,
    },
    # ── Hybrid ────────────────────────────────────────────────────────────
    {
        "label":        "Hybrid_Retraining_02",
        "agent_type":   "RL",
        "base_dir":     os.path.join(TRAINING_ROOT, "Hybrid", "Retraining 2"),
        "prefix":       "Hybrid_Retraining_02_HybridAgent",
        "checkpoints":  [3_499_955, 3_999_944, 4_499_988, 4_999_985, 5_000_113],
        "ckpt_rewards": {
            3_499_955: None,
            3_999_944: None,
            4_499_988: None,
            4_999_985: None,
            5_000_113: None,
        },
        "final_step":   5_000_113,
    },
]

# Bobot composite score untuk agen tipe RL
RL_SCORE_WEIGHTS = {
    "TotalCoverage":    0.50,
    "ExplorationRatio": 0.30,
    "CumulativeReward": 0.20,
}

SMOOTH_WINDOW  = 20
SEARCH_RADIUS  = 300_000   # ±300k steps di sekitar checkpoint
IL_RADIUS      = 200_000   # window lebih kecil untuk IL (data lebih padat)

# ─────────────────────────────────────────────────────────────────────────────
# Fungsi utilitas
# ─────────────────────────────────────────────────────────────────────────────

def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(float(r["Step"])), float(r["Value"])))
            except (ValueError, KeyError):
                pass
    return rows

def moving_avg(data, window):
    result = []
    for i in range(len(data)):
        lo   = max(0, i - window + 1)
        vals = [v for _, v in data[lo:i+1]]
        result.append((data[i][0], sum(vals)/len(vals)))
    return result

def avg_near(data, target, radius):
    near = [v for s, v in data if abs(s - target) <= radius]
    return sum(near)/len(near) if near else None

def normalize_asc(values):
    """Min-max, lebih tinggi = lebih baik."""
    valid = [v for v in values if v is not None]
    if not valid or max(valid) == min(valid):
        return [None if v is None else 0.5 for v in values]
    lo, hi = min(valid), max(valid)
    return [None if v is None else (v - lo)/(hi - lo) for v in values]

def normalize_desc(values):
    """Min-max dibalik, lebih rendah = lebih baik (untuk PolicyLoss)."""
    valid = [v for v in values if v is not None]
    if not valid or max(valid) == min(valid):
        return [None if v is None else 0.5 for v in values]
    lo, hi = min(valid), max(valid)
    return [None if v is None else 1.0 - (v - lo)/(hi - lo) for v in values]

def fmt(v, d=1):
    return f"{v:.{d}f}" if v is not None else "N/A"

def sep(n=68):
    print("=" * n)

# ─────────────────────────────────────────────────────────────────────────────
# Analisis IL
# ─────────────────────────────────────────────────────────────────────────────

def analyse_il(cfg):
    label       = cfg["label"]
    base        = cfg["base_dir"]
    prefix      = cfg["prefix"]
    checkpoints = cfg["checkpoints"]
    final_step  = cfg["final_step"]
    radius      = IL_RADIUS

    sep()
    print(f"  {label}  [tipe: IL — kriteria: PolicyLoss]")
    sep()

    # Muat metrik
    metrics_to_load = ["PolicyLoss", "EpisodeLength", "TotalCoverage", "Entropy"]
    raw    = {}
    smooth = {}
    for name in metrics_to_load:
        # Coba nama file dengan huruf kapital berbeda
        for variant in [name, name.capitalize(), name.upper(), name.lower()]:
            path = os.path.join(base, f"{prefix}_{variant}.csv")
            data = read_csv(path)
            if data:
                raw[name]    = data
                smooth[name] = moving_avg(data, SMOOTH_WINDOW)
                break

    if not raw:
        print("  [!] Tidak ada file CSV ditemukan.")
        return

    all_steps = sorted({s for d in raw.values() for s, _ in d})
    print(f"  Data steps : {min(all_steps):,} — {max(all_steps):,}")
    print(f"  Final ckpt : {final_step:,} steps")

    # Tren PolicyLoss per fase 500k
    print()
    print("  TREN POLICY LOSS per 500k steps")
    print("  (lebih rendah = lebih meniru demo | naik kembali = potensi instabilitas)")
    pl_data = raw.get("PolicyLoss", [])
    if pl_data:
        phases_pl = [(i*500_000, (i+1)*500_000) for i in range(10)]
        prev_avg = None
        for lo, hi in phases_pl:
            vals = [v for s, v in pl_data if lo <= s < hi]
            if not vals:
                continue
            avg = sum(vals)/len(vals)
            if prev_avg is not None:
                delta  = avg - prev_avg
                arrow  = "turun" if delta < -0.0003 else ("NAIK" if delta > 0.0003 else "stabil")
                trend  = f"  {arrow} ({delta:+.4f})"
            else:
                trend = ""
            print(f"    {lo//1000:>4}k–{hi//1000:<4}k : avg={avg:.4f}  min={min(vals):.4f}  max={max(vals):.4f}{trend}")
            prev_avg = avg

    # Nilai per checkpoint
    print()
    print("  NILAI PER CHECKPOINT (avg setiap ±200k steps)")
    print(f"  {'Checkpoint':<14}  {'PolicyLoss':>12}  {'EpLength':>10}  {'TotalCov':>10}  Status")
    print("  " + "-" * 60)

    ckpt_pl = {}
    for ck in checkpoints:
        pl  = avg_near(smooth.get("PolicyLoss",[]), ck, radius)
        el  = avg_near(smooth.get("EpisodeLength",[]), ck, radius)
        cov = avg_near(smooth.get("TotalCoverage",[]), ck, radius)
        ckpt_pl[ck] = pl
        status = "FINAL" if ck == final_step else ""
        print(f"  {ck/1e6:.3f}M steps    {fmt(pl,4):>12}  {fmt(el,0):>10}  {fmt(cov,0):>10}  {status}")

    # Deteksi konvergensi dan kenaikan kembali
    print()
    pl_vals = [(ck, v) for ck, v in ckpt_pl.items() if v is not None]
    if pl_vals:
        best_ck, best_pl = min(pl_vals, key=lambda x: x[1])
        # Cek apakah ada checkpoint setelah best_ck yang lebih tinggi (naik)
        after_best = [(ck, v) for ck, v in pl_vals if ck > best_ck]
        final_worse = any(v > best_pl * 1.03 for _, v in after_best)  # >3% lebih tinggi

        print(f"  PolicyLoss terendah : {best_ck/1e6:.3f}M steps  ({best_pl:.4f})")
        if final_worse:
            worse_steps = [ck for ck, v in after_best if v > best_pl * 1.03]
            print(f"  [!] PolicyLoss naik kembali setelah {best_ck/1e6:.3f}M di step: "
                  + ", ".join(f"{s/1e6:.3f}M" for s in worse_steps))
            print(f"      -> Indikasi overspecialization; gunakan checkpoint {best_ck/1e6:.3f}M")
        else:
            print(f"  PolicyLoss stabil setelah titik terendah — tidak ada kenaikan signifikan.")

        # Rekomendasi
        print()
        print(f"  >> REKOMENDASI  : checkpoint {best_ck/1e6:.3f}M steps")
        print(f"     Alasan: PolicyLoss terendah ({best_pl:.4f})")
        print(f"     Catatan: coverage IL stabil di semua checkpoint (tidak membedakan).")
        if final_worse:
            print(f"     Catatan: final checkpoint TIDAK direkomendasikan (PolicyLoss naik).")

    # Fase training
    print()
    print("  PERKEMBANGAN PER FASE TRAINING")
    phases = [
        ("0–500k",    0,           500_000),
        ("500k–1.5M", 500_000,   1_500_000),
        ("1.5M–3M",   1_500_000, 3_000_000),
        ("3M–4.5M",   3_000_000, 4_500_000),
        ("4.5M–5M",   4_500_000, 5_100_000),
    ]
    pm = ["PolicyLoss", "EpisodeLength", "TotalCoverage"]
    print(f"  {'Fase':<14}", end="")
    for m in pm:
        print(f"  {m:>14}", end="")
    print(f"  {'n':>4}")
    print("  " + "-" * (14 + 16*len(pm) + 6))
    for pname, lo, hi in phases:
        print(f"  {pname:<14}", end="")
        n = 0
        for m in pm:
            vals = [v for s, v in raw.get(m,[]) if lo <= s < hi]
            n = len(vals) if vals else n
            print(f"  {sum(vals)/len(vals):>14.4f}" if vals else f"  {'N/A':>14}", end="")
        print(f"  {n:>4}")

# ─────────────────────────────────────────────────────────────────────────────
# Analisis RL / CDRL / Hybrid
# ─────────────────────────────────────────────────────────────────────────────

def analyse_rl(cfg):
    label       = cfg["label"]
    base        = cfg["base_dir"]
    prefix      = cfg["prefix"]
    checkpoints = cfg["checkpoints"]
    final_step  = cfg["final_step"]
    aliases     = cfg.get("metric_aliases", {})   # nama file alternatif per metrik
    note        = cfg.get("note", "")

    sep()
    label_note = f"  [{note}]" if note else ""
    print(f"  {label}  [tipe: RL — kriteria: Coverage + ExplRatio + Reward]{label_note}")
    sep()

    all_metric_names = [
        "TotalCoverage", "ExplorationRatio", "CumulativeReward",
        "ExtrinsicReward", "CuriosityReward", "EpisodeLength",
        "GoldenPathCount", "ExplorationCount",
    ]
    raw    = {}
    smooth = {}
    for name in all_metric_names:
        # Coba nama standar dulu, lalu alias jika ada
        file_name = aliases.get(name, name)
        data = read_csv(os.path.join(base, f"{prefix}_{file_name}.csv"))
        if data:
            raw[name]    = data
            smooth[name] = moving_avg(data, SMOOTH_WINDOW)

    if not raw:
        print("  [!] Tidak ada file CSV ditemukan.")
        return

    all_steps = sorted({s for d in raw.values() for s, _ in d})
    print(f"  Data steps : {min(all_steps):,} — {max(all_steps):,}")
    print(f"  Final ckpt : {final_step:,} steps")
    if max(all_steps) < final_step:
        gap = final_step - max(all_steps)
        print(f"  [!] TensorBoard berhenti {gap:,} steps sebelum final checkpoint")
        print(f"      -> Final checkpoint TIDAK DAPAT dibandingkan")

    # Tabel nilai per checkpoint
    print()
    shown = [n for n in all_metric_names if n in smooth]
    print(f"  {'Checkpoint':<14}", end="")
    for n in shown:
        print(f"  {n[:14]:>14}", end="")
    print(f"  {'Status':>14}")
    print("  " + "-" * (14 + 16*len(shown) + 16))

    ckpt_data = {}
    for ck in checkpoints:
        row    = {}
        status = "FINAL" if ck == final_step else ""
        if max(all_steps) < ck - SEARCH_RADIUS:
            status += " (no TB data)"
        print(f"  {ck/1e6:.3f}M steps", end="")
        for name in shown:
            v = avg_near(smooth[name], ck, SEARCH_RADIUS) if name in smooth else None
            row[name] = v
            print(f"  {fmt(v):>14}", end="")
        ckpt_data[ck] = row
        print(f"  {status:>14}")

    # Composite score
    print()
    print("  COMPOSITE SCORE  (Coverage 50% + ExplRatio 30% + Reward 20%)")
    print("  Catatan: coverage dan exploration ratio lebih langsung")
    print("  mencerminkan kemampuan jelajah di Map B daripada reward.")
    print(f"  {'Checkpoint':<14}  {'Cov(norm)':>10}  {'Expl(norm)':>10}  {'Rew(norm)':>10}  {'SCORE':>8}  {'Rank':>5}")
    print("  " + "-" * 68)

    score_metrics = list(RL_SCORE_WEIGHTS.keys())
    raw_vals  = {m: [ckpt_data[ck].get(m) for ck in checkpoints] for m in score_metrics}
    norm_vals = {m: normalize_asc(raw_vals[m]) for m in score_metrics}

    scores = []
    for i, ck in enumerate(checkpoints):
        total, valid = 0.0, True
        for m, w in RL_SCORE_WEIGHTS.items():
            nv = norm_vals[m][i]
            if nv is None:
                valid = False
                break
            total += nv * w
        scores.append((ck, total if valid else None))

    rankable = sorted([(ck, s) for ck, s in scores if s is not None], key=lambda x: -x[1])
    rank_map  = {ck: r+1 for r, (ck, _) in enumerate(rankable)}

    for i, ck in enumerate(checkpoints):
        _, sc   = scores[i]
        cov_n   = norm_vals["TotalCoverage"][i]
        expl_n  = norm_vals["ExplorationRatio"][i]
        rew_n   = norm_vals["CumulativeReward"][i]
        rank    = rank_map.get(ck, "-")
        marker  = " <-- TERBAIK" if rank == 1 else ""
        print(
            f"  {ck/1e6:.3f}M steps"
            f"  {fmt(cov_n,3):>10}"
            f"  {fmt(expl_n,3):>10}"
            f"  {fmt(rew_n,3):>10}"
            f"  {fmt(sc,3):>8}"
            f"  {str(rank):>5}{marker}"
        )

    if rankable:
        best_ck, best_sc = rankable[0]
        print()
        print(f"  >> REKOMENDASI  : checkpoint {best_ck/1e6:.3f}M steps")
        print(f"     Composite score : {best_sc:.3f}")
        if best_ck != final_step:
            print(f"     Catatan: bukan final checkpoint.")
        if max(all_steps) < final_step:
            print(f"     Catatan: final checkpoint tidak dapat dinilai (data TB tidak lengkap).")

    # ── Analisis peak TotalCoverage sepanjang training ─────────────────────
    if "TotalCoverage" in smooth:
        tc_smooth = smooth["TotalCoverage"]
        peak_step, peak_val = max(tc_smooth, key=lambda x: x[1])
        final_val = avg_near(tc_smooth, final_step, SEARCH_RADIUS)
        print()
        print(f"  ANALISIS PEAK TOTAL COVERAGE (kurva penuh)")
        print(f"    Peak    : step {peak_step/1e6:.3f}M  coverage={peak_val:.1f}")
        print(f"    Final   : step {final_step/1e6:.3f}M  coverage={fmt(final_val)}")
        if final_val is not None and peak_val > 0:
            pct_diff = (peak_val - final_val) / peak_val * 100
            if peak_step < final_step and pct_diff > 5:
                print(f"    [!] Coverage turun {pct_diff:.1f}% dari peak ke final")
                print(f"        -> Model terbaik seharusnya checkpoint ~{peak_step/1e6:.1f}M, bukan final")
            elif peak_step == final_step or pct_diff <= 5:
                print(f"    [OK] Coverage di final masih dalam 5% dari peak — final cukup optimal")

    # Fase training
    print()
    print("  PERKEMBANGAN PER FASE TRAINING")
    phases = [
        ("BC/Early (0–500k)",   0,           500_000),
        ("PPO1 (500k–2M)",      500_000,   2_000_000),
        ("PPO2 (2M–3.5M)",      2_000_000, 3_500_000),
        ("PPO3 (3.5M–5M)",      3_500_000, 5_100_000),
    ]
    pm = ["TotalCoverage", "ExplorationRatio", "CumulativeReward", "EpisodeLength"]
    print(f"  {'Fase':<22}", end="")
    for m in pm:
        print(f"  {m[:14]:>14}", end="")
    print(f"  {'n':>4}")
    print("  " + "-" * (22 + 16*len(pm) + 6))
    for pname, lo, hi in phases:
        print(f"  {pname:<22}", end="")
        n = 0
        for m in pm:
            vals = [v for s, v in raw.get(m,[]) if lo <= s < hi]
            n = len(vals) if vals else n
            print(f"  {sum(vals)/len(vals):>14.1f}" if vals else f"  {'N/A':>14}", end="")
        print(f"  {n:>4}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

for cfg in CONFIGS:
    print()
    if cfg.get("agent_type") == "IL":
        analyse_il(cfg)
    else:
        analyse_rl(cfg)

print()
sep()
print("RINGKASAN REKOMENDASI FINAL")
sep()
print(f"  {'Agen':<25}  {'Checkpoint':>12}  Kriteria seleksi")
print("  " + "-" * 65)
print(f"  {'IL_Training_03':<25}  {'4.000M steps':>12}  PolicyLoss terendah (early stopping)")
print(f"  {'CDRL_Retraining_03':<25}  {'5.000M steps':>12}  Coverage+ExplRatio+Reward terus naik")
print(f"  {'Hybrid_Retraining_02':<25}  {'3.500M steps':>12}  Coverage tertinggi (final tidak terdata)")
print()
sep()
print("Selesai.")
