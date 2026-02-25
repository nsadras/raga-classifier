import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
from tqdm import tqdm

from preprocessing import compute_melodic_score


logger = logging.getLogger(__name__)


def load_manifest(csv_path: str) -> pd.DataFrame:
    logger.info(f"Loading manifest from {csv_path}")
    df = pd.read_csv(csv_path)
    expected_cols = {"Song Name", "Ragam", "Composer", "Youtube Link", "Song Take"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    # Normalize column names we use downstream
    df = df.rename(columns={
        "Song Name": "song_name",
        "Ragam": "raga",
        "Composer": "composer",
        "Youtube Link": "youtube_link",
        "Song Take": "song_take",
        "Janya Number": "janya_number",
    })
    # Extract youtube id
    df["ytid"] = df["youtube_link"].apply(extract_ytid)
    n_rows = len(df)
    n_ragas = df["raga"].nunique()
    missing_ytid = df["ytid"].isna().sum()
    logger.info(f"Loaded {n_rows} rows across {n_ragas} ragas; missing ytid: {missing_ytid}")
    return df


def extract_ytid(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    # Common patterns: https://www.youtube.com/watch?v=VIDEOID[&...], https://youtu.be/VIDEOID[?...] 
    # Quick regex attempts
    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{6,})",
        r"youtu\.be/([A-Za-z0-9_-]{6,})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{6,})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _ensure_ffmpeg_available() -> None:
    # yt-dlp relies on ffmpeg to postprocess to wav
    from shutil import which
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg to enable audio extraction.")


def download_youtube_audio(
    youtube_url: str,
    out_wav_path: Path,
    cookies_from_browser: Optional[str] = None,
    browser_profile: Optional[str] = None,
    cookies_file: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Download YouTube audio and convert to mono wav using ffmpeg via yt-dlp.
    Returns (success, error_message)
    """
    try:
        _ensure_ffmpeg_available()
        import yt_dlp
    except Exception as e:  # pragma: no cover
        return False, f"Dependencies missing: {e}"

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a temp template; we'll ensure final is out_wav_path
    outtmpl = str(out_wav_path.with_suffix("").parent / (out_wav_path.stem + ".%(ext)s"))
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        # Enforce mono during conversion
        "postprocessor_args": ["-ac", "1"],
        "quiet": False,
        "noprogress": False,
        "ignoreerrors": False,
        "noplaylist": True,
        "retries": 3,
    }
       
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
        logger.info(f"Using cookies from file '{cookies_file}'")
    elif cookies_from_browser:
        # Use cookies from a local browser to bypass consent/age walls
        if browser_profile:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser, browser_profile)
            logger.info(f"Using cookies from browser '{cookies_from_browser}' profile '{browser_profile}'")
        else:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
            logger.info(f"Using cookies from browser '{cookies_from_browser}' (default profile)")
    try:
        logger.info(f"Downloading audio → {youtube_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        # yt-dlp created a .wav with same stem
        produced = out_wav_path
        if not produced.exists():
            # try to find the produced wav
            candidate = list(out_wav_path.parent.glob(out_wav_path.stem + "*.wav"))
            if candidate:
                candidate[0].rename(out_wav_path)
        if out_wav_path.exists():
            logger.info(f"Saved → {out_wav_path}")
            return True, None
        else:
            logger.warning(f"yt-dlp did not produce wav for {youtube_url}")
            return False, "wav not produced"
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        return False, str(e)


def download_audio_for_manifest(
    df: pd.DataFrame,
    raw_dir: Path,
    max_per_raga: Optional[int] = None,
    cookies_from_browser: Optional[str] = None,
    browser_profile: Optional[str] = None,
    cookies_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Iterate over manifest and download audio if missing. Respects max_per_raga if provided.
    Adds columns: local_path, dl_ok, dl_error
    """
    logger.info(f"Starting downloads to {raw_dir} (cap per raga: {max_per_raga})")
    counts: Dict[str, int] = {}
    rows: List[dict] = []
    total = len(df)
    processed = 0
    success = 0
    failed = 0
    skipped_cap = 0
    skipped_missing = 0
    for _, row in tqdm(df.iterrows(), total=total, desc="Downloading audio"):
        raga = row["raga"]
        ytid = row.get("ytid")
        link = row.get("youtube_link")
        if not isinstance(ytid, str) or len(ytid) < 6:
            rows.append({**row, "local_path": None, "dl_ok": False, "dl_error": "missing_ytid"})
            skipped_missing += 1
            continue
        if max_per_raga is not None:
            if counts.get(raga, 0) >= max_per_raga:
                rows.append({**row, "local_path": None, "dl_ok": False, "dl_error": "skipped_by_cap"})
                skipped_cap += 1
                continue

        out_path = raw_dir / f"{ytid}.wav"
        if out_path.exists():
            rows.append({**row, "local_path": str(out_path), "dl_ok": True, "dl_error": None})
            counts[raga] = counts.get(raga, 0) + 1
            success += 1
            processed += 1
            continue

        ok, err = download_youtube_audio(
            link,
            out_path,
            cookies_from_browser=cookies_from_browser,
            browser_profile=browser_profile,
            cookies_file=cookies_file,
        )
        rows.append({**row, "local_path": str(out_path) if ok else None, "dl_ok": ok, "dl_error": err})
        if ok:
            counts[raga] = counts.get(raga, 0) + 1
            success += 1
        else:
            failed += 1
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total} (ok={success}, failed={failed}, skipped_cap={skipped_cap}, missing_ytid={skipped_missing})")

    new_df = pd.DataFrame(rows)
    logger.info(f"Finished downloads: ok={success}, failed={failed}, skipped_cap={skipped_cap}, missing_ytid={skipped_missing}")
    return new_df


def _load_wav_segment(path: str, start_sec: float, end_sec: float) -> Tuple[torch.Tensor, int]:
    info = sf.info(path)
    sr = info.samplerate
    start_frame = int(start_sec * sr)
    # Ensure end_frame does not exceed total frames
    end_frame = min(int(end_sec * sr), info.frames)
    
    # Calculate frames to read
    frames_to_read = end_frame - start_frame
    if frames_to_read <= 0:
         return torch.zeros(0), sr
         
    data, _ = sf.read(path, start=start_frame, stop=end_frame, always_2d=False)
    
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    waveform = torch.from_numpy(data)
    return waveform, sr


class MertAudioDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        audio_root: Path,
        target_sample_rate: int,
        segment_seconds: float = 20.0,
        segment_hop_seconds: float = 10.0,
        strategy: str = "simple",  # "simple" or "smart"
        min_melodic_score: float = -np.inf,
        max_segments_per_file: Optional[int] = None,
        index_cache_path: Optional[Path] = None,
    ) -> None:
        self.manifest = manifest.copy()
        self.audio_root = Path(audio_root)
        self.target_sample_rate = int(target_sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_hop_seconds = float(segment_hop_seconds)
        self.strategy = strategy
        self.min_melodic_score = min_melodic_score
        self.max_segments_per_file = max_segments_per_file
        self.index_cache_path = Path(index_cache_path) if index_cache_path else None

        self.index: List[Tuple[int, float, float]] = []  # (row_idx, start_sec, end_sec)
        self.resamplers = {}
        self._build_index()

    def _build_index(self) -> None:
        import pickle
        if self.index_cache_path and self.index_cache_path.exists():
            logger.info(f"Loading dataset index from {self.index_cache_path}")
            try:
                with open(self.index_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Verify config matches
                    if cached_data.get('strategy') == self.strategy and \
                       cached_data.get('min_melodic_score') == self.min_melodic_score and \
                       cached_data.get('segment_seconds') == self.segment_seconds and \
                       cached_data.get('segment_hop_seconds') == self.segment_hop_seconds: # Corrected key
                        self.index = cached_data['index']
                        logger.info(f"Loaded {len(self.index)} segments from cache.")
                        return
                    else:
                        logger.warning("Cached index config mismatch, rebuilding...")
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}")

        files = 0
        for i, row in tqdm(self.manifest.iterrows(), total=len(self.manifest), desc="Building dataset index"):
            local_path = row.get("local_path")
            dl_ok = row.get("dl_ok", False)
            if not dl_ok or not isinstance(local_path, str) or not Path(local_path).exists():
                continue
            files += 1
            
            # Load full waveform if smart strategy is needed, otherwise just probe duration
            try:
                if self.strategy == "smart":
                    # We need the waveform to score segments. 
                    # Use _load_wav_segment(..., 0, None) -> no, helper expects end_sec.
                    # Helper above computes frames if we give seconds?
                    # For full file reading in smart mode, we have to read all.
                    # Reimplement efficient full read here or use helper with large duration?
                    # Use sf.read directly for full file here.
                    full_wav_np, sr = sf.read(local_path, always_2d=False)
                    if full_wav_np.ndim == 2: full_wav_np = full_wav_np.mean(axis=1)
                    if full_wav_np.dtype != np.float32: full_wav_np = full_wav_np.astype(np.float32)
                    full_wav = torch.from_numpy(full_wav_np)
                    duration_sec = full_wav.shape[0] / sr
                else:
                    with sf.SoundFile(local_path) as f:
                        duration_sec = len(f) / f.samplerate
            except Exception as e:
                logger.warning(f"Error reading {local_path}: {e}")
                continue

            seg_len = self.segment_seconds
            hop = self.segment_hop_seconds
            
            candidates = []
            if duration_sec < max(seg_len, 1.0):
                candidates.append((0.0, min(duration_sec, seg_len)))
            else:
                start = 0.0
                while start + 1e-6 < duration_sec:
                    end = min(start + seg_len, duration_sec)
                    candidates.append((start, end))
                    if end >= duration_sec:
                        break
                    start += hop
            
            # Filter/Select candidates
            selected = []
            if self.strategy == "smart":
                scored_candidates = []
                for start, end in candidates:
                    s_idx = int(start * sr)
                    e_idx = int(end * sr)
                    # Use full_wav from memory
                    segment_wav = full_wav[s_idx:e_idx].numpy()
                    score = compute_melodic_score(segment_wav, sr)
                    if score >= self.min_melodic_score:
                        scored_candidates.append((score, start, end))
                
                # Sort by score descending
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Take top N
                if self.max_segments_per_file:
                    scored_candidates = scored_candidates[:self.max_segments_per_file]
                
                selected = [(s, e) for _, s, e in scored_candidates]
            else:
                # Simple strategy
                if self.max_segments_per_file:
                    selected = candidates[:self.max_segments_per_file]
                else:
                    selected = candidates

            for start, end in selected:
                self.index.append((i, start, end))

        logger.info(f"Dataset index built: {files} files, {len(self.index)} segments (strategy={self.strategy})")
        
        # Save cache if path provided
        if self.index_cache_path:
            import pickle
            try:
                cache_data = {
                    'strategy': self.strategy,
                    'min_melodic_score': self.min_melodic_score,
                    'segment_seconds': self.segment_seconds,
                    'segment_hop_seconds': self.segment_hop_seconds,
                    'index': self.index
                }
                with open(self.index_cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Saved dataset index cache to {self.index_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save index cache: {e}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        row_idx, start_sec, end_sec = self.index[idx]
        row = self.manifest.iloc[row_idx]
        path = row["local_path"]
        
        # Optimized load: read ONLY the segment from disk
        waveform, sr = _load_wav_segment(path, start_sec, end_sec)

        if waveform.numel() == 0:
             logger.warning(f"Loaded empty waveform for {path} (start={start_sec}, end={end_sec}). Skipping.")
             return None

        # Resample if needed
        if sr != self.target_sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = T.Resample(sr, self.target_sample_rate)
            waveform = self.resamplers[sr](waveform.unsqueeze(0)).squeeze(0)

        info = {
            "song_name": row.get("song_name"),
            "raga": row.get("raga"),
            "composer": row.get("composer"),
            "ytid": row.get("ytid"),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "sampling_rate": int(self.target_sample_rate),
            "local_path": str(path),
        }
        return waveform, info


