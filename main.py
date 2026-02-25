import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import datetime
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
    get_audio_encoder,
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
    parser.add_argument("--model", type=str, default="m-a-p/MERT-v1-330M", help="HF model id for MERT or checkpoint for CLAP")
    parser.add_argument("--model_type", type=str, default="mert", choices=["mert", "clap", "culturemert"], help="Model architecture")
    parser.add_argument("--strategy", type=str, default="simple", choices=["simple", "smart"], help="Segment selection strategy")
    parser.add_argument("--min_melodic_score", type=float, default=1.0, help="Min score for smart selection")
    parser.add_argument("--cookies_from_browser", type=str, default=None, help="Pass a browser name to use cookies (e.g., 'chrome', 'safari', 'firefox')")
    parser.add_argument("--cookies_file", type=str, default=None, help="Path to cookies.txt/netscape cookies file")
    parser.add_argument("--browser_profile", type=str, default=None, help="Optional browser profile name (e.g., 'Default')")
    parser.add_argument("--task", type=str, default="quickcheck", choices=["quickcheck", "umap", "cache_all_layers", "cache_segments", "analyze_cached_segments", "benchmark"], help="Task to run")
    parser.add_argument("--layer", type=int, default=12, help="MERT layer index [0..24] for UMAP")
    parser.add_argument("--max_segments", type=int, default=0, help="If >0, randomly sample at most this many segments for embedding")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for segment sampling")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for embedding generation")
    parser.add_argument("--index_cache", type=str, default="data/index_cache.pkl", help="Path to cache dataset index")
    parser.add_argument("--results_dir", type=str, default=None, help="Specific output directory (overrides timestamp generation)")
    args = parser.parse_args()

    data_root = Path(args.out_dir)
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.results_dir:
        base_dir = Path(args.results_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        base_dir = Path(f"results-{timestamp}")
    
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {base_dir}")

    logger.info("Loading manifest and starting downloads…")
    df = load_manifest(args.csv)
    df = download_audio_for_manifest(
        df,
        raw_dir,
        max_per_raga=None if args.max_per_raga <= 0 else args.max_per_raga,
        cookies_from_browser=args.cookies_from_browser,
        browser_profile=args.browser_profile,
        cookies_file=args.cookies_file,
    )

    # Handle aliases
    real_model_type = args.model_type
    if args.model_type == "culturemert":
        real_model_type = "mert"
        if args.model == "m-a-p/MERT-v1-330M": # User didn't override default
            args.model = "ntua-slp/CultureMERT-95M"

    logger.info(f"Loading model ({args.model_type} -> {real_model_type}): {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")
    
    # Speed optimizations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.benchmark = True

    model, processor = get_audio_encoder(real_model_type, args.model, device)
    
    # Determine SR
    target_sr = 16000 # default & forced for MERT to align with benchmark speed
    if real_model_type == "clap":
        target_sr = 48000
    
    # (Ignored processor SR to ensure 16k consistency which was proven fast)
    if processor and real_model_type != "clap" and hasattr(processor, "sampling_rate") and processor.sampling_rate != 16000:
        logger.info(f"Overriding processor SR {processor.sampling_rate} with 16000 for speed consistency.")
    
    # Determine SR (default 16000 unless CLAP is primarily specified or needed)
    # For benchmark, we might need a unifying SR or load dataset to compatible SR.
    # MERT/CultureMERT needs 16000/24000 (usually 24k). CLAP needs 48000.
    # If we use one dataset instance, we must pick one SR.
    # But different models need different SRs.
    # MertAudioDataset does on-the-fly resampling if we ask it to?
    # No, it loads at target_sample_rate and that's it.
    # If we want to support both efficiently, we might need two datasets or on-the-fly resampling in collate.
    # But for now, let's just create the dataset with 48k (CLAP) and let MERT processor downsample?
    # MERT processor downsamples? No, usually expects 16k/24k.
    # Ideally, we should instantiate dataset per model or changing SR.
    # BUT we want to cache the index. The index depends on SR for start/end samples?
    # The index stores (start_sec, end_sec) which is SR-independent.
    # However, smart selection computes score on waveform.
    # If we cache the index, we can reuse it with different SRs!
    
    # Let's handle benchmark specific logic inside the task usually, but dataset init is here.
    # We will Init dataset dynamically in benchmark task.
    
    if args.task == "benchmark":
        # Run comparison
        models_to_run = [
            ("mert", "m-a-p/MERT-v1-330M"),
            ("culturemert", "ntua-slp/CultureMERT-95M"),
            ("clap", "laion/clap-htsat-unfused"),
        ]
        
        # We need to build/cache index once. 
        # Pick a robust SR for indexing (e.g. 16000)
        cache_path = Path(args.index_cache)
        
        # Ensure raw dir populated
        # (already done above)

        logger.info("Starting Benchmark Comparison...")
        results_csv = base_dir / "benchmark_results.csv"
        if results_csv.exists(): results_csv.unlink()

        from analysis import compute_clustering_metrics
        import gc

        for m_type, m_id in models_to_run:
            logger.info(f"--- Benchmarking {m_type} ({m_id}) ---")
            
            # Determine SR for this model
            if m_type == "clap":
                sr = 48000
            else:
                sr = 16000 # MERT/CultureMERT usually 16k or 24k. 24k is better for MERT v1? 16k is safe default.
            
            # Init dataset (will reuse cache if compatible config)
            # CAUTION: If cache stores sample INDICES, it depends on SR.
            # My raga_data previous edit: index is (row_idx, start_sec, end_sec).
            # And smart selection computes score.
            # The cache saves strategy params. It DOES NOT save SR?
            # Wait, `_build_index` uses `sr` to slice: `s_idx = int(start * sr)`.
            # If `start` and `end` are seconds, then it is fine!
            # The cache stores `self.index`. `self.index` has (row_idx, start_sec, end_sec).
            # So YES, we can reuse cache across SRs!
            
            # Update: raga_data verifies check: `target_sample_rate` is NOT in the cache check I wrote.
            # Good.
            
            ds = MertAudioDataset(
                manifest=df,
                audio_root=raw_dir,
                target_sample_rate=sr,
                segment_seconds=args.segment_seconds,
                segment_hop_seconds=args.segment_hop,
                strategy=args.strategy,
                min_melodic_score=args.min_melodic_score,
                index_cache_path=cache_path
            )
            
            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, processor = get_audio_encoder(
                "mert" if m_type == "culturemert" else m_type, 
                m_id, 
                device
            )
            
            # Compute
            indices = None
            if args.max_segments > 0:
                random.seed(args.sample_seed)
                # We need consistent sampling across models for fair comparison?
                # Benchmark usually runs full? Or same subset.
                # If we use same seed and range, should be same subset if dataset len is same.
                indices = list(range(len(ds)))
                random.shuffle(indices)
                indices = indices[: args.max_segments]

            emb_df = compute_track_embeddings(
                ds, model, processor, 
                layer_index=args.layer,
                dataset_indices=indices,
                model_type="mert" if m_type == "culturemert" else m_type,
                batch_size=args.batch_size
            )
            
            # Metrics
            for color_col in ["raga", "janya_number"]:
                if color_col not in emb_df.columns: continue
                sub_df = emb_df[emb_df[color_col].notna()]
                if len(sub_df) < 2: continue
                
                X = np.stack(sub_df["embedding"].values)
                labels = sub_df[color_col].astype(str).values
                scores = compute_clustering_metrics(X, labels)
                logger.info(f"{m_type} [{color_col}]: {scores}")
                
                res_row = {"model": m_type, "label": color_col, **scores}
                pd.DataFrame([res_row]).to_csv(results_csv, mode='a', header=not results_csv.exists(), index=False)
            
            # Cleanup
            del model, processor, ds, emb_df
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"Benchmark complete. Results saved to {results_csv}")
        return

    # For other tasks, we need legacy dataset init
    if args.task == "quickcheck":
        logger.info("Task is quickcheck: limiting dataset to first 5 files for speed.")
        df = df.head(5)

    dataset = MertAudioDataset(
        manifest=df,
        audio_root=raw_dir,
        target_sample_rate=target_sr,
        segment_seconds=args.segment_seconds,
        segment_hop_seconds=args.segment_hop,
        strategy=args.strategy,
        min_melodic_score=args.min_melodic_score,
        index_cache_path=args.index_cache if args.task != "quickcheck" else None
    )

    if len(dataset) == 0:
        logger.warning(f"No audio segments available using strategy={args.strategy}. Try lowering min score or checking data.")
        return

    if args.task == "quickcheck":
        waveform, info = dataset[0]
        logger.info(f"Processing one sample for sanity check: {info['ytid']} {info['song_name']} ({info['raga']})")
        
        if real_model_type == "mert":
            inputs = processor(waveform, sampling_rate=target_sr, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
            logger.info(f"MERT Hidden states: {all_layer_hidden_states.shape}")
        elif real_model_type == "clap":
            # HF transformers CLAP
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.numpy()
            else:
                waveform_np = waveform
            
            inputs = processor(audios=waveform_np, sampling_rate=48000, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.get_audio_features(**inputs)
            logger.info(f"CLAP Embedding: {outputs.shape}")
            
    elif args.task == "umap":
        logger.info(f"Computing track embeddings (Model: {args.model_type})")
        indices = None
        if args.max_segments and args.max_segments > 0:

            random.seed(args.sample_seed)
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[: args.max_segments]
            logger.info(f"Sampling {len(indices)} segments (seed={args.sample_seed})")
            
        emb_df = compute_track_embeddings(
            dataset, model, processor, 
            layer_index=args.layer, 
            dataset_indices=indices,
            model_type=real_model_type,
            batch_size=args.batch_size
        )
        plots_dir = base_dir / "plots"
        prefix = f"umap_{args.model_type}"
        if real_model_type == "mert":
            prefix += f"_layer{args.layer}"
            
        # Export Interactive JSON
        from analysis import export_interactive_data
        json_path = base_dir / "plots" / "umap_interactive.json"
        export_interactive_data(emb_df, json_path)
            
        logger.info("Interactive data exported. Open 'http://localhost:8000' to view.")
        
        from analysis import compute_clustering_metrics
        results_csv = base_dir / "comparison_results.csv"
        
        for color_col in ["raga", "janya_number"]:
            if color_col not in emb_df.columns: continue
            
            # Filter NaNs
            sub_df = emb_df[emb_df[color_col].notna()]
            if len(sub_df) < 2: continue
            
            # Embeddings matrix
            X = np.stack(sub_df["embedding"].values)
            labels = sub_df[color_col].astype(str).values
            
            scores = compute_clustering_metrics(X, labels)
            logger.info(f"Scores for {args.model_type} by {color_col}: {scores}")
            
            # Save to CSV
            res_row = {
                "model": args.model_type,
                "layer": args.layer if real_model_type == "mert" else "last",
                "label": color_col,
                **scores
            }
            res_df = pd.DataFrame([res_row])
            res_df.to_csv(results_csv, mode='a', header=not results_csv.exists(), index=False)

            umap_project_and_plot(emb_df, color_col=color_col, out_html=plots_dir / f"{prefix}_by_{color_col}.html")
    elif args.task == "cache_all_layers":
        # cache_all_layers
        cache_dir = base_dir / "embeddings"
        indices = None
        if args.max_segments and args.max_segments > 0:

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
