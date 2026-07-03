# CDRLILHybrid

Peningkatan Kemampuan Generalisasi Agen Playtester Hybrid Imitation Learning dan Curiosity-Driven Reinforcement Learning pada Lingkungan Baru (Unseen Environment)

Proyek skripsi yang membandingkan tiga arsitektur Unity ML-Agents untuk deteksi bug otomatis lewat AI playtesting:

- **Pure Imitation Learning (IL)**
- **Pure Curiosity-Driven Reinforcement Learning (CDRL)**
- **Hybrid IL + CDRL**

Ketiga agen dilatih di **Map A** (urban) dan diuji generalisasinya di **Map B** (forest, ~80x80 unit, 1161 grid NavMesh valid) tanpa retraining.

## Daftar Isi

1. [Tech Stack](#tech-stack)
2. [Struktur Repository](#struktur-repository)
3. [Prasyarat](#prasyarat)
4. [Instalasi](#instalasi)
5. [Melatih Agen](#melatih-agen)
6. [Menjalankan Inference / Playtesting](#menjalankan-inference--playtesting)
7. [Script Analisis](#script-analisis)
8. [Metrik Evaluasi](#metrik-evaluasi)
9. [Data & Output di Google Drive](#data--output-di-google-drive)

## Tech Stack

| Komponen | Versi |
|---|---|
| Unity Editor | 6000.2.8f1 |
| Unity ML-Agents (package) | 1.1.0 (embedded di `Packages/com.unity.ml-agents@...`) |
| Python | 3.10 |
| mlagents / mlagents-envs | 1.1.0 |
| PyTorch | 2.10.0 (CPU) |
| protobuf | 3.20.3 |
| numpy | 1.23.5 |

Daftar lengkap dependency Python ada di [`requirements.txt`](requirements.txt).

## Struktur Repository

```
AITestingEnvironment/
├── Assets/
│   ├── Scripts/AIPlayerControls/   # CDRLAgent.cs, ILAgent.cs, HybridAgent.cs, RLAgent.cs
│   ├── Scripts/Environment/        # InferenceController, GridCounter, GeometryBugMarker, dll.
│   ├── Scenes/                     # Map A.unity, Map B.unity
│   ├── Models/                     # checkpoint .onnx hasil training
│   └── GoldenPath/                 # referensi jalur golden path
├── Config/                         # config training ML-Agents (.yaml)
├── scripts/                        # script analisis Python (lihat bagian Script Analisis)
├── ProjectSettings/, Packages/     # konfigurasi & dependency Unity
└── requirements.txt                # dependency Python
```

CSV hasil playtest (`BugReport*.csv`, `CoverageData*.csv`, `PositionLog*.csv`, `InferenceLog*.csv`), folder `results/` (output training ML-Agents), dan `Assets/_Recovery/` **tidak** disimpan di git — lihat [Data & Output di Google Drive](#data--output-di-google-drive).

## Prasyarat

- **Unity Hub** + **Unity Editor 6000.2.8f1** (tambahkan lewat Unity Hub > Installs kalau belum ada)
- **Python 3.10** (versi lain berisiko konflik dependency dengan `mlagents==1.1.0`)
- **Git**
- Windows dengan **Command Prompt (cmd)** — proyek ini didokumentasikan untuk cmd, bukan PowerShell

## Instalasi

### 1. Clone repository

```
git clone https://github.com/Spoopage/CDRLILHybrid.git
cd CDRLILHybrid
```

### 2. Buka project di Unity

Buka Unity Hub, pilih **Add project from disk**, arahkan ke folder hasil clone, lalu buka dengan Editor **6000.2.8f1**. Unity akan otomatis resolve semua package di `Packages/manifest.json`, termasuk package ML-Agents lokal (`com.unity.ml-agents@...`) yang sudah di-embed di repo — tidak perlu install ulang lewat Package Manager.

### 3. Setup environment Python

Buat virtual environment (nama konvensinya `~venv`, tapi nama apa pun boleh):

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Verifikasi ML-Agents terpasang dengan benar:

```
mlagents-learn --help
```

### 4. Siapkan folder demonstrasi IL (khusus training IL/Hybrid)

Config `Config/pure_il_config.yaml` dan `Config/hybrid_il_cdrl_config.yaml` membaca file demonstrasi (`.demo`) dari path `demo_path` yang di-hardcode ke lokasi lokal penulis asli. Sebelum training IL/Hybrid, sesuaikan `demo_path` di kedua file tersebut ke lokasi file `.demo` hasil rekam kamu sendiri (direkam lewat komponen **Demonstration Recorder** di Unity Editor pada agen terkait).

## Melatih Agen

Training dijalankan lewat `mlagents-learn`, lalu tekan Play di Unity Editor saat diminta untuk menyambungkan environment.

```
venv\Scripts\activate
mlagents-learn Config\pure_cdrl_config.yaml --run-id=CDRL_Run1
mlagents-learn Config\pure_il_config.yaml --run-id=IL_Run1
mlagents-learn Config\hybrid_il_cdrl_config.yaml --run-id=Hybrid_Run1
```

Config lain yang tersedia di `Config/`:

| Config | Behavior | Catatan |
|---|---|---|
| `pure_cdrl_config.yaml` | `CDRLAgent` | PPO + curiosity reward signal |
| `pure_il_config.yaml` | `ILAgent` | PPO + behavioral cloning dari demo |
| `hybrid_il_cdrl_config.yaml` | `HybridAgent` | PPO + curiosity + behavioral cloning |
| `pure_rl_config.yaml` | `RLAgent` | Baseline PPO murni tanpa curiosity (dipakai untuk pengujian awal, bukan salah satu dari 3 arsitektur utama skripsi) |

Buka scene `Assets/Scenes/Map A.unity` sebelum training (agen dilatih di Map A). Checkpoint `.onnx` akan tersimpan otomatis ke `results/<run-id>/` selama training; hasil akhirnya disalin manual ke `Assets/Models/` untuk dipakai saat inference.

Catatan penting saat mengedit script agen (`CDRLAgent.cs`, `ILAgent.cs`, `HybridAgent.cs`):
- Gunakan `CompletedEpisodes` bawaan ML-Agents untuk hitung episode, jangan pakai counter manual (pernah menyebabkan `episodeCount` inflasi).
- Pertahankan signature method lifecycle ML-Agents (`OnEpisodeBegin`, `CollectObservations`, `OnActionReceived`).

## Menjalankan Inference / Playtesting

Untuk generalization test di **Map B**:

1. Buka scene `Assets/Scenes/Map B.unity`.
2. Assign model `.onnx` yang sudah dipilih (lihat [checkpoint_selector.py](#script-analisis)) ke Behavior Parameters agen terkait, set ke mode **Inference Only**.
3. Cek komponen `InferenceController` di scene — default 100 episode, `timeScale` 5, tapi nilai aktual biasanya di-override lewat Inspector, bukan dari default di kode.
4. Jalankan Play. `InferenceController` akan menulis log lewat polling di `Update()` (bukan event-based) untuk keandalan logging.

Output tersimpan sebagai CSV di `Assets/` (`BugReport_*.csv`, `CoverageData_*.csv`, `PositionLog_*.csv`, `InferenceLog_*.csv`) — file-file ini di-gitignore, pindahkan ke Google Drive setelah run selesai.

## Script Analisis

Semua script analisis ada di [`scripts/`](scripts/) dan dijalankan dengan Python (venv yang sama seperti training):

| Script | Fungsi |
|---|---|
| `checkpoint_selector.py` | Membandingkan checkpoint training (berdasarkan TotalCoverage, ExplorationRatio, CumulativeReward untuk RL; PolicyLoss untuk IL) dan merekomendasikan checkpoint terbaik |
| `analisis_mapb_lokal.py` | Analisis metrik lengkap Map A + Map B: coverage, generalization gap, precision/recall/F1, AUC, efisiensi, saturasi — plus semua chart untuk laporan |
| `analisis_fp_candidates.py` | Mengumpulkan dan mengelompokkan (clustering spasial) kandidat false positive dari ketiga agen untuk verifikasi manual di Unity Editor |

### Konfigurasi lokasi data

Ketiga script membaca CSV hasil training/inference dari sebuah folder data eksternal (di luar repo git, biasanya folder sinkronisasi Google Drive bernama `SKRIPSI`). Lokasi ini dikontrol lewat environment variable:

```
set SKRIPSI_DATA_ROOT=D:\path\ke\folder\SKRIPSI
```

Kalau tidak di-set, script fallback ke default `C:\Kuliah\Skripsi\SKRIPSI`. Folder ini diharapkan berisi struktur:

```
SKRIPSI/
├── Data Training/<CDRL|IL|Hybrid>/...TotalCoverage.csv, dll.
├── Data Pengujian 100 eps/<CDRL|IL|Hybrid>/BugReport_*.csv, CoverageData_*.csv, dll.
└── Hasil Pengujian/                # output chart & CSV hasil analisis ditulis ke sini
```

### Menjalankan

```
venv\Scripts\activate
set SKRIPSI_DATA_ROOT=D:\path\ke\folder\SKRIPSI
python scripts\checkpoint_selector.py
python scripts\analisis_mapb_lokal.py
python scripts\analisis_fp_candidates.py
```

Urutan pemakaian yang disarankan: `checkpoint_selector.py` dulu untuk memilih checkpoint per agen, lalu jalankan inference di Map B pakai checkpoint terpilih, baru jalankan `analisis_mapb_lokal.py` untuk metrik utama dan `analisis_fp_candidates.py` untuk menyaring kandidat ground truth baru dari false positive yang berulang.

## Metrik Evaluasi

Coverage, generalization gap, precision/recall/F1, AUC, dan efisiensi grid/menit. Analisis bersifat deskriptif-komparatif, tanpa uji statistik inferensial.

Ground truth: 12 lokasi bug, diverifikasi post-hoc dengan mengurutkan frekuensi deteksi lalu verifikasi manual di Unity Editor.

Konvensi threshold:
- Threshold deteksi (jarak ke ground truth): 2 unit — catatan, titik ground truth N3 dan N5 berjarak ~0.427 unit sehingga rawan tertukar.
- Threshold PhysicsBug: 300f absolut.

## Data & Output di Google Drive

Data mentah (CSV hasil training/inference dalam jumlah besar) dan graf hasil analisis disimpan di Google Drive, bukan di repo ini:

**[\[link Google Drive\]](https://drive.google.com/drive/folders/16GkaGvXRK93G6gUyXU3XyfzYpz0rjvKu?usp=sharing)**
