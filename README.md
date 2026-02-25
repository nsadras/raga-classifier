# Raga Classifier

Exploring raga classification using audio foundation models (MERT, CultureMERT, CLAP). Audio segments are embedded into a high-dimensional feature space and visualized with UMAP, with clustering quality scored by silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.

## Motivation

Ragas are the fundamental melodic frameworks of Indian classical music. Automatically identifying them from audio is a hard problem: ragas share many notes and ornaments, and recordings vary enormously in duration and style. Large self-supervised audio models pre-trained on diverse music offer strong priors that may generalize to this domain without requiring large labeled datasets.

This project asks: **do MERT and CultureMERT representations form raga-separable clusters in embedding space?** UMAP plots and clustering metrics let us quickly compare models and layers.

## Processing Pipeline

```
CSV Manifest
    │
    ▼
Download Audio (yt-dlp + ffmpeg → mono WAV)
    │
    ▼
Segment Index (sliding window, simple or smart melodic filtering)
    │
    ▼
Audio Embeddings (MERT / CultureMERT / CLAP, per-segment)
    │
    ▼
Track-level Pooling (mean over segments per track)
    │
    ▼
UMAP Projection + Clustering Metrics
    │
    ▼
Interactive Viewer (served locally via server.py)
```

**Segment strategies:**
- `simple` — fixed-size sliding window (default 20 s, 10 s hop)
- `smart` — same sliding window, but segments are scored by melodic content (pitch clarity) and only those above `--min_melodic_score` are kept; top-scoring segments are preferred

## Requirements

- Python 3.11+
- `ffmpeg` on PATH (for audio extraction via yt-dlp)
- A YouTube cookies file (`cookies.txt`) for age-restricted or region-locked content
- GPU recommended; CPU works but is slow

## Setup

```bash
# Install dependencies (uses uv)
uv sync
```

## Quick Start

Sanity-check the pipeline on 5 songs:

```bash
uv run python main.py \
  --csv CarnaticSongsDatabase.csv \
  --task quickcheck \
  --max_per_raga 5
```

## Full Analysis

Run both MERT and CultureMERT UMAP analyses on the full dataset:

```bash
./run_full_analysis.sh
```

Results are written to a timestamped `results-YYYY-MM-DD-HH-MM/` directory.

## Interactive Dashboard

The dashboard lets you explore UMAP plots and listen to audio clips by clicking on points.

### 1. Start the server

```bash
uv run python server.py
```

Then open **http://localhost:8000** in your browser.

### 2. Select a run

The index page lists all `results-*/` directories that contain a UMAP plot. Click any entry to open the interactive viewer for that run.

You can also link directly to a specific JSON file:

```
http://localhost:8000/viewer.html?file=results-2026-01-21-14-07/plots/umap_interactive.json
```

### 3. Use the viewer

- **Click any point** on the UMAP plot to load and play the corresponding audio segment (starts at the exact segment offset, not the beginning of the file).
- **Color By** dropdown — switch between coloring points by raga name or janya number.
- The audio player at the bottom supports standard playback controls (play/pause, volume, seeking within the segment).

> **Note:** The server must be running for audio playback to work, as it serves the `.wav` files from `data/raw/`. Seeking in the audio player requires the server to support HTTP range requests, which `server.py` handles via `RangeHTTPServer`.

## Manual Runs

### UMAP (track-level embeddings)

```bash
uv run python main.py \
  --task umap \
  --model_type mert \           # mert | culturemert | clap
  --strategy smart \            # simple | smart
  --min_melodic_score -4.0 \    # lower = less filtering
  --batch_size 16 \
  --cookies_file cookies.txt \
  --results_dir my_results
```

### Cache All-Layer Embeddings (for deeper analysis)

```bash
uv run python main.py \
  --task cache_all_layers \
  --model_type mert \
  --results_dir my_results
```

Then analyze the cached embeddings across all 25 MERT layers:

```bash
uv run python main.py \
  --task analyze_cached_segments \
  --results_dir my_results
```

### Benchmark (compare all models)

```bash
uv run python main.py \
  --task benchmark \
  --strategy smart \
  --min_melodic_score -4.0 \
  --batch_size 16 \
  --cookies_file cookies.txt \
  --results_dir benchmark_results
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | `CarnaticSongsDatabase.csv` | Path to song manifest CSV |
| `--out_dir` | `data` | Base directory for downloaded audio |
| `--results_dir` | auto-timestamped | Where to save outputs |
| `--model_type` | `mert` | `mert`, `culturemert`, or `clap` |
| `--task` | `quickcheck` | `quickcheck`, `umap`, `benchmark`, `cache_all_layers`, `cache_segments`, `analyze_cached_segments` |
| `--strategy` | `simple` | Segment selection: `simple` or `smart` |
| `--min_melodic_score` | `1.0` | Minimum melodic score (smart strategy); lower = less filtering |
| `--segment_seconds` | `20.0` | Length of each audio segment |
| `--segment_hop` | `10.0` | Hop between segments |
| `--layer` | `12` | MERT hidden layer to use for UMAP |
| `--batch_size` | `16` | Batch size for embedding generation |
| `--max_per_raga` | `10` | Max tracks to download per raga |
| `--cookies_file` | — | Path to Netscape-format cookies file |
| `--cookies_from_browser` | — | Browser to extract cookies from (e.g. `chrome`) |
