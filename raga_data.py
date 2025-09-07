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
        "quiet": True,
        "noprogress": True,
        "ignoreerrors": True,
        "retries": 3,
    }
    if cookies_from_browser:
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


def _load_wav_mono(path: str) -> Tuple[torch.Tensor, int]:
    data, sr = sf.read(path, always_2d=False)
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
    ) -> None:
        self.manifest = manifest.copy()
        self.audio_root = Path(audio_root)
        self.target_sample_rate = int(target_sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_hop_seconds = float(segment_hop_seconds)

        self.index: List[Tuple[int, float, float]] = []  # (row_idx, start_sec, end_sec)
        self._build_index()

    def _build_index(self) -> None:
        files = 0
        for i, row in self.manifest.iterrows():
            local_path = row.get("local_path")
            dl_ok = row.get("dl_ok", False)
            if not dl_ok or not isinstance(local_path, str) or not Path(local_path).exists():
                continue
            files += 1
            # Probe duration via soundfile; avoid loading here by reading header
            try:
                with sf.SoundFile(local_path) as f:
                    duration_sec = len(f) / f.samplerate
            except Exception:
                continue

            seg_len = self.segment_seconds
            hop = self.segment_hop_seconds
            if duration_sec < max(seg_len, 1.0):
                # Take a single segment covering available audio
                self.index.append((i, 0.0, min(duration_sec, seg_len)))
                continue

            start = 0.0
            while start + 1e-6 < duration_sec:
                end = min(start + seg_len, duration_sec)
                self.index.append((i, start, end))
                if end >= duration_sec:
                    break
                start += hop
        logger.info(f"Dataset index built: {files} files, {len(self.index)} segments")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        row_idx, start_sec, end_sec = self.index[idx]
        row = self.manifest.iloc[row_idx]
        path = row["local_path"]
        waveform, sr = _load_wav_mono(path)

        # Trim to segment
        s = int(start_sec * sr)
        e = int(end_sec * sr)
        e = min(e, waveform.shape[0])
        segment = waveform[s:e]

        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = T.Resample(sr, self.target_sample_rate)
            segment = resampler(segment.unsqueeze(0)).squeeze(0)

        info = {
            "song_name": row.get("song_name"),
            "raga": row.get("raga"),
            "composer": row.get("composer"),
            "ytid": row.get("ytid"),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "sampling_rate": int(self.target_sample_rate),
        }
        return segment, info


