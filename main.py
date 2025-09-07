import argparse
import logging
import os
from pathlib import Path

from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch

from raga_data import load_manifest, download_audio_for_manifest, MertAudioDataset
from analysis import (
    compute_track_embeddings,
    umap_project_and_plot,
    compute_and_cache_track_embeddings,
    compute_and_cache_segment_embeddings,
    analyze_cached_segments,
    load_cached_embeddings,
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Raga classifier data prep and analysis")
    parser.add_argument("--csv", type=str, default="CarnaticSongsDatabase.csv", help="Path to CSV manifest")
    parser.add_argument("--out_dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--max_per_raga", type=int, default=10, help="Limit downloads per raga (0 for all)")
    parser.add_argument("--segment_seconds", type=float, default=20.0, help="Segment length in seconds")
    parser.add_argument("--segment_hop", type=float, default=10.0, help="Segment hop in seconds")
    parser.add_argument("--model", type=str, default="m-a-p/MERT-v1-330M", help="HF model id for MERT")
    parser.add_argument("--cookies_from_browser", type=str, default=None, help="Pass a browser name to use cookies (e.g., 'chrome', 'safari', 'firefox')")
    parser.add_argument("--browser_profile", type=str, default=None, help="Optional browser profile name (e.g., 'Default')")
    parser.add_argument("--task", type=str, default="quickcheck", choices=["quickcheck", "umap", "cache_all_layers", "cache_segments", "analyze_cached_segments"], help="Task to run")
    parser.add_argument("--layer", type=int, default=12, help="MERT layer index [0..24] for UMAP")
    parser.add_argument("--max_segments", type=int, default=0, help="If >0, randomly sample at most this many segments for embedding")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for segment sampling")
    args = parser.parse_args()

    base_dir = Path(args.out_dir)
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading manifest and starting downloads…")
    df = load_manifest(args.csv)
    df = download_audio_for_manifest(
        df,
        raw_dir,
        max_per_raga=None if args.max_per_raga <= 0 else args.max_per_raga,
        cookies_from_browser=args.cookies_from_browser,
        browser_profile=args.browser_profile,
    )

    logger.info(f"Loading model and processor: {args.model}")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model, trust_remote_code=True)

    dataset = MertAudioDataset(
        manifest=df,
        audio_root=raw_dir,
        target_sample_rate=processor.sampling_rate,
        segment_seconds=args.segment_seconds,
        segment_hop_seconds=args.segment_hop,
    )

    if len(dataset) == 0:
        logger.warning("No audio available after download.")
        return

    if args.task == "quickcheck":
        waveform, info = dataset[0]
        logger.info(f"Processing one sample for sanity check: {info['ytid']} {info['song_name']} ({info['raga']})")
        inputs = processor(waveform, sampling_rate=processor.sampling_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
        logger.info(f"Hidden states (25 x 1024 expected): {time_reduced_hidden_states.shape}")
    elif args.task == "umap":
        logger.info(f"Computing track embeddings at layer {args.layer}")
        indices = None
        if args.max_segments and args.max_segments > 0:
            import random
            random.seed(args.sample_seed)
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[: args.max_segments]
            logger.info(f"Sampling {len(indices)} segments (seed={args.sample_seed})")
        emb_df = compute_track_embeddings(dataset, model, processor, layer_index=args.layer, dataset_indices=indices)
        plots_dir = base_dir / "plots"
        umap_project_and_plot(emb_df, color_col="raga", out_html=plots_dir / f"umap_layer{args.layer}_by_raga.html")
        umap_project_and_plot(emb_df, color_col="janya_number", out_html=plots_dir / f"umap_layer{args.layer}_by_janya.html")
    elif args.task == "cache_all_layers":
        # cache_all_layers
        cache_dir = base_dir / "embeddings"
        indices = None
        if args.max_segments and args.max_segments > 0:
            import random
            random.seed(args.sample_seed)
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[: args.max_segments]
            logger.info(f"Sampling {len(indices)} segments (seed={args.sample_seed}) for caching")
        compute_and_cache_track_embeddings(dataset, model, processor, out_dir=cache_dir, dataset_indices=indices)
    elif args.task == "cache_segments":
        cache_dir = base_dir / "embeddings"
        indices = None
        if args.max_segments and args.max_segments > 0:
            import random
            random.seed(args.sample_seed)
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[: args.max_segments]
            logger.info(f"Sampling {len(indices)} segments (seed={args.sample_seed}) for segment caching")
        compute_and_cache_segment_embeddings(dataset, model, processor, out_dir=cache_dir, dataset_indices=indices)
    elif args.task == "analyze_cached_segments":
        cache_dir = base_dir / "embeddings"
        npz_path = cache_dir / "segment_embeddings_all_layers.npz"
        meta_csv = cache_dir / "segment_metadata.csv"
        if not npz_path.exists() or not meta_csv.exists():
            logger.error("Cached segment embeddings not found. Run --task cache_segments first.")
            return
        analyze_cached_segments(npz_path=npz_path, meta_csv=meta_csv, out_dir=base_dir)


if __name__ == "__main__":
    main()
