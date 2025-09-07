import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import umap
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from raga_data import MertAudioDataset


logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_track_embeddings(
    dataset: MertAudioDataset,
    model: AutoModel,
    processor: Wav2Vec2FeatureExtractor,
    device: Optional[torch.device] = None,
    layer_index: int = 12,
    per_track_pool: str = "mean",
    dataset_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns: ytid, raga, janya_number, song_name, embedding (np.ndarray)
    Aggregates segment embeddings into one track-level embedding by mean.
    layer_index in [0..24].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    track_to_segments: Dict[str, List[np.ndarray]] = {}
    meta_rows: Dict[str, Dict] = {}

    indices = dataset_indices if dataset_indices is not None else list(range(len(dataset)))
    logger.info(f"Embedding {len(indices)} segment(s) out of {len(dataset)} total")
    for i in tqdm(indices, desc="Embedding segments"):
        waveform, info = dataset[i]
        inputs = processor(waveform, sampling_rate=processor.sampling_rate, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states: tuple length 25, each [B,T,1024]
        hs = outputs.hidden_states[layer_index].squeeze(0)  # [T, 1024]
        seg_embed = hs.mean(dim=0)  # [1024]
        ytid = info.get("ytid")
        if ytid not in track_to_segments:
            track_to_segments[ytid] = []
            meta_rows[ytid] = {
                "ytid": ytid,
                "raga": info.get("raga"),
                "song_name": info.get("song_name"),
                "janya_number": None,
            }
        track_to_segments[ytid].append(seg_embed.detach().cpu().numpy())

    # Attach janya_number if present in dataset.manifest
    if "Janya Number" in dataset.manifest.columns:
        jmap = dataset.manifest.set_index("ytid")["Janya Number"].to_dict()
    elif "janya_number" in dataset.manifest.columns:
        jmap = dataset.manifest.set_index("ytid")["janya_number"].to_dict()
    else:
        jmap = {}
    for ytid, meta in meta_rows.items():
        meta["janya_number"] = jmap.get(ytid)

    rows = []
    for ytid, seg_list in track_to_segments.items():
        arr = np.stack(seg_list, axis=0)  # [S, 1024]
        if per_track_pool == "mean":
            track_embed = arr.mean(axis=0)
        elif per_track_pool == "median":
            track_embed = np.median(arr, axis=0)
        else:
            raise ValueError("Unsupported pool: " + per_track_pool)
        rows.append({**meta_rows[ytid], "embedding": track_embed})

    df = pd.DataFrame(rows)
    logger.info(f"Computed track embeddings: {len(df)} tracks at layer {layer_index}")
    return df


def umap_project_and_plot(
    df_emb: pd.DataFrame,
    color_col: str,
    out_html: Path,
    random_state: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    metric: str = "cosine",
) -> None:
    X = np.stack(df_emb["embedding"].values)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    X2 = reducer.fit_transform(Xs)
    plot_df = df_emb.copy()
    plot_df["umap_x"] = X2[:, 0]
    plot_df["umap_y"] = X2[:, 1]
    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color=color_col,
        hover_data=["ytid", "song_name", "raga", "janya_number"],
        title=f"UMAP projection colored by {color_col}",
        width=900,
        height=700,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    logger.info(f"Saved plot → {out_html}")


@torch.no_grad()
def compute_and_cache_track_embeddings(
    dataset: MertAudioDataset,
    model: AutoModel,
    processor: Wav2Vec2FeatureExtractor,
    out_dir: Path,
    device: Optional[torch.device] = None,
    dataset_indices: Optional[List[int]] = None,
) -> Tuple[Path, Path]:
    """
    Computes track-level embeddings for ALL 25 layers, pooling segments by mean.
    Saves two files in out_dir:
      - embeddings npz: 'track_embeddings_all_layers.npz' with array (N, 25, 1024)
      - metadata csv:   'track_metadata.csv' with columns [ytid, raga, janya_number, song_name]
    Returns paths to (npz_path, metadata_csv).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    indices = dataset_indices if dataset_indices is not None else list(range(len(dataset)))
    logger.info(f"Embedding (all layers) for {len(indices)} segment(s) out of {len(dataset)} total")

    # Accumulators per track
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    meta_rows: Dict[str, Dict] = {}

    for idx in tqdm(indices, desc="Embedding segments (all layers)"):
        waveform, info = dataset[idx]
        inputs = processor(waveform, sampling_rate=processor.sampling_rate, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        # Stack to [25, T, 1024]
        hs_all = torch.stack(outputs.hidden_states).squeeze(1)  # [25, T, 1024]
        seg_embed_all = hs_all.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [25, 1024]
        ytid = info.get("ytid")

        if ytid not in sums:
            sums[ytid] = np.zeros_like(seg_embed_all, dtype=np.float64)
            counts[ytid] = 0
            meta_rows[ytid] = {
                "ytid": ytid,
                "raga": info.get("raga"),
                "song_name": info.get("song_name"),
                "janya_number": info.get("janya_number"),
            }
        sums[ytid] += seg_embed_all
        counts[ytid] += 1

    # Build stable order by ytid
    ytids = sorted(sums.keys())
    N = len(ytids)
    if N == 0:
        raise RuntimeError("No embeddings computed; dataset may be empty.")

    emb = np.zeros((N, seg_embed_all.shape[0], seg_embed_all.shape[1]), dtype=np.float32)
    rows = []
    for i, ytid in enumerate(ytids):
        emb[i] = (sums[ytid] / max(counts[ytid], 1)).astype(np.float32)
        m = meta_rows[ytid]
        rows.append({
            "ytid": m.get("ytid"),
            "raga": m.get("raga"),
            "janya_number": m.get("janya_number"),
            "song_name": m.get("song_name"),
        })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "track_embeddings_all_layers.npz"
    meta_csv = out_dir / "track_metadata.csv"
    # Save embeddings and simple row order index
    np.savez_compressed(npz_path, embeddings=emb, ytids=np.array(ytids, dtype=object))
    pd.DataFrame(rows).to_csv(meta_csv, index=False)
    logger.info(f"Saved embeddings → {npz_path} (shape={emb.shape})")
    logger.info(f"Saved metadata   → {meta_csv}")
    return npz_path, meta_csv


def load_cached_embeddings(npz_path: Path, meta_csv: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"]  # (N, 25, 1024)
    meta = pd.read_csv(meta_csv)
    return emb, meta


@torch.no_grad()
def compute_and_cache_segment_embeddings(
    dataset: MertAudioDataset,
    model: AutoModel,
    processor: Wav2Vec2FeatureExtractor,
    out_dir: Path,
    device: Optional[torch.device] = None,
    dataset_indices: Optional[List[int]] = None,
) -> Tuple[Path, Path]:
    """
    Computes segment-level embeddings for ALL 25 layers (mean over time per segment).
    Saves two files in out_dir:
      - embeddings npz: 'segment_embeddings_all_layers.npz' with array (S, 25, 1024)
      - metadata csv:   'segment_metadata.csv' with [segment_idx, ytid, raga, janya_number, song_name, start_sec, end_sec]
    Returns paths to (npz_path, metadata_csv).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    indices = dataset_indices if dataset_indices is not None else list(range(len(dataset)))
    logger.info(f"Embedding segments (all layers): {len(indices)} of {len(dataset)} total")

    # Map janya from manifest if present
    if "janya_number" in dataset.manifest.columns:
        jmap = dataset.manifest.set_index("ytid")["janya_number"].to_dict()
    elif "Janya Number" in dataset.manifest.columns:
        jmap = dataset.manifest.set_index("ytid")["Janya Number"].to_dict()
    else:
        jmap = {}

    seg_embeds: List[np.ndarray] = []
    meta_rows: List[Dict] = []

    for idx in tqdm(indices, desc="Embedding segments (all layers)"):
        waveform, info = dataset[idx]
        inputs = processor(waveform, sampling_rate=processor.sampling_rate, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hs_all = torch.stack(outputs.hidden_states).squeeze(1)  # [25, T, 1024]
        seg_embed_all = hs_all.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [25, 1024]
        seg_embeds.append(seg_embed_all)
        ytid = info.get("ytid")
        meta_rows.append({
            "segment_idx": idx,
            "ytid": ytid,
            "raga": info.get("raga"),
            "janya_number": jmap.get(ytid),
            "song_name": info.get("song_name"),
            "start_sec": info.get("start_sec"),
            "end_sec": info.get("end_sec"),
        })

    emb = np.stack(seg_embeds, axis=0)  # (S, 25, 1024)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "segment_embeddings_all_layers.npz"
    meta_csv = out_dir / "segment_metadata.csv"
    np.savez_compressed(npz_path, embeddings=emb)
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
    logger.info(f"Saved segment embeddings → {npz_path} (shape={emb.shape})")
    logger.info(f"Saved segment metadata   → {meta_csv}")
    return npz_path, meta_csv


def umap_plot_from_matrix(
    X: np.ndarray,
    meta: pd.DataFrame,
    color_col: str,
    out_html: Path,
    title_prefix: Optional[str] = None,
    random_state: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    metric: str = "cosine",
) -> None:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    X2 = reducer.fit_transform(Xs)
    plot_df = meta.copy()
    plot_df["umap_x"] = X2[:, 0]
    plot_df["umap_y"] = X2[:, 1]
    title = f"UMAP by {color_col}" if not title_prefix else f"{title_prefix} | by {color_col}"
    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color=color_col,
        hover_data=["ytid", "song_name", "raga", "janya_number"],
        title=title,
        width=900,
        height=700,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    logger.info(f"Saved plot → {out_html}")


def analyze_cached_segments(
    npz_path: Path,
    meta_csv: Path,
    out_dir: Path,
    color_cols: Optional[List[str]] = None,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    metric: str = "cosine",
) -> Path:
    if color_cols is None:
        color_cols = ["raga", "janya_number"]
    X_all, meta = load_cached_embeddings(npz_path, meta_csv)
    # X_all: (S, 25, 1024)
    num_layers = X_all.shape[1]
    results: List[Dict] = []
    plots_root = Path(out_dir) / "plots" / "segments"

    for color_col in color_cols:
        if color_col not in meta.columns:
            logger.warning(f"Column '{color_col}' not found in metadata; skipping")
            continue
        # Filter to rows with valid labels
        valid_mask = meta[color_col].notna()
        if valid_mask.sum() < 2:
            logger.warning(f"Not enough labeled segments for '{color_col}'; skipping")
            continue
        meta_sub = meta.loc[valid_mask].reset_index(drop=True)
        X_sub = X_all[valid_mask.values]

        for layer in range(num_layers):
            X_layer = X_sub[:, layer, :]
            labels = meta_sub[color_col].astype(str).values
            sil = np.nan
            db = np.nan
            ch = np.nan

            # Compute silhouette in the original embedding space
            try:
                sil = float(silhouette_score(X_layer, labels, metric=metric))
            except Exception as e:
                logger.warning(f"Silhouette failed for layer {layer} by '{color_col}': {e}")

            # Compute DB and CH on standardized embedding space (Euclidean)
            try:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X_layer)
                db = float(davies_bouldin_score(Xs, labels))
                ch = float(calinski_harabasz_score(Xs, labels))
            except Exception as e:
                logger.warning(f"DB/CH failed for layer {layer} by '{color_col}': {e}")

            parts = []
            parts.append(f"Layer {layer}")
            parts.append(f"sil={sil:.3f}" if sil == sil else "sil=nan")
            parts.append(f"DB={db:.3f}" if db == db else "DB=nan")
            parts.append(f"CH={ch:.1f}" if ch == ch else "CH=nan")
            title = " | ".join(parts)
            out_html = plots_root / f"seg_umap_layer{layer:02d}_by_{color_col}.html"
            try:
                umap_plot_from_matrix(
                    X_layer,
                    meta_sub,
                    color_col=color_col,
                    out_html=out_html,
                    title_prefix=title,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                )
            except Exception as e:
                logger.warning(f"UMAP plot failed for layer {layer} by '{color_col}': {e}")
            results.append({"layer": layer, "label": color_col, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})

    scores_path = Path(out_dir) / "silhouette_scores_segments.csv"
    pd.DataFrame(results).to_csv(scores_path, index=False)
    logger.info(f"Saved silhouette scores → {scores_path}")
    return scores_path


