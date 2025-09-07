# raga classifier
Exploring raga classification using audio foundation models

## Quick start

Prepare data and run a quick MERT check:

```bash
uv run python main.py --csv CarnaticSongsDatabase.csv --out_dir data --max_per_raga 5
```

Note: ffmpeg must be installed and on PATH for audio extraction.
