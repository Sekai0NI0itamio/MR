# MR – Music Recognizer 🎵

**GitHub Actions-powered music fingerprint recognition.**
Upload audio files, trigger a workflow, and get the top 20 matching tracks with confidence scores — all without leaving GitHub.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Quick Start](#quick-start)
3. [Repository Structure](#repository-structure)
4. [Adding Audio Files](#adding-audio-files)
5. [Running the Workflow](#running-the-workflow)
6. [Understanding the Results](#understanding-the-results)
7. [Artifacts & Logs](#artifacts--logs)
8. [Configuration](#configuration)
9. [Database Management](#database-management)
10. [Troubleshooting](#troubleshooting)
11. [Extending MR](#extending-mr)
12. [Testing](#testing)
13. [License & Attribution](#license--attribution)

---

## How It Works

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User adds   │     │  GitHub Actions   │     │   Download      │
│  audio files │────▶│  workflow starts  │────▶│   artifacts     │
│  to incoming/│     │  (manual trigger) │     │   (results ZIP) │
└──────────────┘     └──────────────────┘     └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              ┌──────────┐ ┌──────┐ ┌──────────┐
              │ Normalize│ │Finger│ │  Search   │
              │  (ffmpeg)│ │print │ │  (FAISS)  │
              │          │ │(fpcalc)│ │          │
              └──────────┘ └──────┘ └──────────┘
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                     ┌────────────────┐
                     │  Top 20 matches│
                     │  per file +    │
                     │  summary table │
                     └────────────────┘
```

### Pipeline (per file)

1. **Normalize** – Convert audio to mono 16-bit 44.1 kHz WAV with `ffmpeg`.
2. **Fingerprint** – Generate a Chromaprint fingerprint with `fpcalc`.
3. **Search** – Query a FAISS index built from the [AcoustID](https://acoustid.org/) database for the 20 nearest fingerprint matches.
4. **Result** – Write a JSON file with track title, artist, album, and confidence score (0–1) for each match.

### Data Source

The recognition database comes from the [AcoustID project](https://acoustid.org/). Database dumps are available at **https://data.acoustid.org/** under the **CC BY-SA 3.0** license. The first workflow run downloads and indexes the data; subsequent runs use a cached copy.

---

## Quick Start

### 1. Fork / Clone

```bash
git clone https://github.com/<your-user>/MR.git
cd MR
```

### 2. Add audio files

Drop one or more `.mp3`, `.m4a`, or `.ogg` files into the `incoming/` folder:

```bash
cp ~/Music/mystery-track.mp3 incoming/
git add incoming/
git commit -m "Add audio files for recognition"
git push
```

### 3. Run the workflow

1. Go to your repository on GitHub.
2. Click **Actions** → **🎵 Recognize Music** → **Run workflow**.
3. (Optional) Set the number of parallel workers.
4. Click **Run workflow**.

### 4. Download results

Once the workflow completes:
1. Go to the workflow run page.
2. Download the **mr-results** artifact (ZIP).
3. Open `results/<filename>.json` for the top 20 matches.

---

## Repository Structure

```
MR/
├── .github/
│   └── workflows/
│       ├── recognize.yml       # Main recognition workflow (manual trigger)
│       ├── ci.yml              # Unit tests on push/PR
│       └── update-db.yml       # Weekly database refresh (scheduled)
├── incoming/                   # ← Drop audio files here
├── mr/                         # Python package
│   ├── __init__.py
│   ├── __main__.py             # python -m mr entry point
│   ├── audio.py                # Audio normalisation & fingerprinting
│   ├── config.py               # Paths, thresholds, knobs
│   ├── db.py                   # FAISS index building & querying
│   ├── models.py               # TrackMatch / RecognitionResult dataclasses
│   ├── pipeline.py             # Orchestrator (concurrent processing)
│   ├── search.py               # Top-K search logic
│   ├── summary.py              # GitHub Actions summary generation
│   └── utils.py                # Logging & file discovery
├── scripts/
│   ├── download_db.py          # Download AcoustID CSV dumps
│   └── build_index.py          # Build FAISS index from CSVs
├── tests/                      # Unit & integration tests
│   ├── conftest.py
│   ├── test_audio.py
│   ├── test_config.py
│   ├── test_db.py
│   ├── test_integration.py
│   ├── test_models.py
│   ├── test_search.py
│   ├── test_summary.py
│   └── test_utils.py
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Test / dev dependencies
├── README.md                   # ← You are here
└── .gitignore
```

---

## Adding Audio Files

Place files in the `incoming/` directory. Supported formats:

| Format | Extension |
|--------|-----------|
| MP3    | `.mp3`    |
| M4A    | `.m4a`    |
| OGG    | `.ogg`    |

**Tips:**
- You can add multiple files in one commit; they'll all be processed in a single workflow run.
- Filenames can contain spaces or special characters, but keeping them simple makes results easier to find.
- Large files (>25 MB) may need [Git LFS](https://git-lfs.github.com/) since GitHub has a per-file size limit.

```bash
# Example: add several files at once
cp ~/Music/*.mp3 incoming/
git add incoming/
git commit -m "Add tracks for recognition"
git push
```

---

## Running the Workflow

1. Navigate to **Actions** in your GitHub repository.
2. Select **🎵 Recognize Music** from the left sidebar.
3. Click **Run workflow** (top right).
4. Configure optional inputs:

| Input | Default | Description |
|-------|---------|-------------|
| `max_workers` | `4` | Number of files to process concurrently |
| `rebuild_index` | `false` | Force re-download and rebuild of the AcoustID database |

5. Click the green **Run workflow** button.

The workflow will:
- Check out the repo
- Install Python, `ffmpeg`, and `fpcalc`
- Restore or build the fingerprint database
- Process each audio file through the pipeline
- Upload a ZIP artifact with results, logs, and debug files
- Print a summary table in the GitHub Actions step summary

---

## Understanding the Results

### JSON output (per file)

Each input file produces a JSON file in the artifact under `results/<stem>.json`:

```json
{
  "input_file": "mystery-track.mp3",
  "duration_seconds": 242.5,
  "matches": [
    {
      "track_id": "abc123",
      "title": "Bohemian Rhapsody",
      "artist": "Queen",
      "album": "A Night at the Opera",
      "confidence": 0.9821
    },
    {
      "track_id": "def456",
      "title": "Another Match",
      "artist": "Some Artist",
      "album": "Some Album",
      "confidence": 0.7134
    }
  ],
  "status": "success",
  "error": null
}
```

### Confidence scores

| Range | Meaning |
|-------|---------|
| **0.8 – 1.0** | Strong match — very likely the correct track |
| **0.5 – 0.8** | Moderate match — plausible but verify manually |
| **< 0.5** | Weak match — included for completeness; treat with caution |

The system always returns up to 20 matches regardless of score. Matches with confidence ≥ 0.5 are counted as "high confidence" in the summary table.

### Summary table

The **Actions run page** shows a Markdown summary:

| Filename | Duration (s) | Matches (≥0.5) | Top Match | Confidence | Status | Error |
|----------|-------------|----------------|-----------|------------|--------|-------|
| track.mp3 | 242.5 | 12 | Bohemian Rhapsody – Queen | 0.9821 | ✅ success | |
| broken.ogg | 0.0 | 0 | — | — | ❌ failed | ffmpeg failed: Invalid data |

---

## Artifacts & Logs

The workflow uploads a single artifact named **mr-results** containing:

```
mr-results/
├── results/          # One JSON per input file (see above)
│   ├── track1.json
│   └── track2.json
├── logs/             # Detailed logs
│   ├── pipeline.log  # Master pipeline log
│   ├── track1.log    # Per-file processing log
│   └── track2.log
└── debug/            # Intermediate files (optional)
    ├── track1_normalised.wav
    ├── track1_fingerprint.json
    └── ...
```

Artifacts are retained for **30 days** by default.

---

## Configuration

All configuration is in `mr/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MR_PROJECT_ROOT` | repo root | Base path for all directories |
| `MR_TOP_K` | `20` | Number of matches to return per file |
| `MR_HIGH_CONF` | `0.5` | Threshold for "high confidence" count |
| `MR_MAX_WORKERS` | `4` | Concurrent processing threads |
| `MR_FP_URL` | AcoustID URL | Fingerprint CSV download URL |
| `MR_TRACK_URL` | AcoustID URL | Track metadata CSV download URL |

In the workflow, set these via the `env:` block or as workflow inputs.

---

## Database Management

### First run

On the first workflow execution (or when no cache exists), the pipeline will:
1. Download the AcoustID fingerprint and track CSV dumps (~3 GB compressed).
2. Build a FAISS index from the fingerprint data.
3. Cache both in the GitHub Actions cache for future runs.

### Automatic updates

The **🔄 Update Database** workflow runs weekly (Sunday 04:00 UTC) and can also be triggered manually. It:
1. Downloads the latest AcoustID dumps.
2. Rebuilds the FAISS index.
3. Saves the new index to the cache.

### Manual rebuild

To force a rebuild during recognition, set `rebuild_index` to `true` when triggering the workflow.

### Cache key

The cache is keyed on the hash of `scripts/download_db.py` and `scripts/build_index.py`. If you change these scripts, the cache will automatically invalidate.

---

## Troubleshooting

### "No audio files found in incoming/"

You haven't committed any supported audio files. Ensure:
- Files are in `incoming/` (not a subdirectory).
- Extensions are `.mp3`, `.m4a`, or `.ogg`.
- Files are committed and pushed to the branch you're running the workflow on.

### "Failed to load fingerprint database"

The FAISS index hasn't been built yet. This usually means:
- The first run's database download timed out (the AcoustID dump is large).
- The cache was evicted. Re-run the workflow or trigger **🔄 Update Database**.

### A file shows ❌ failed

Check the per-file log in `logs/<filename>.log` inside the artifact. Common causes:
- **Corrupted audio** – `ffmpeg` can't decode the file.
- **Too short** – `fpcalc` needs a minimum audio duration (~2 seconds).
- **Unsupported codec** – the container is `.mp3` but the codec isn't recognized.

### Workflow times out

The free GitHub Actions runner has 6 hours max. If the database download or indexing takes too long:
- Use `--max-rows` in the scripts to limit the dataset size.
- Consider using a GitHub-hosted larger runner or a self-hosted runner.

### Cache not restoring

Caches are scoped to the branch. If you're on a feature branch, the cache from `main` might not restore. Push to `main` first or trigger the database update workflow on your branch.

---

## Extending MR

### Replacing the recognition backend

The search is decoupled from the audio processing:

1. **Audio → fingerprint**: Modify `mr/audio.py` to use a different fingerprinting tool (e.g., `dejavu`, `audfprint`).
2. **Fingerprint → matches**: Modify `mr/db.py` and `mr/search.py` to use a different index (SQLite with `sqlite-vec`, Annoy, PostgreSQL with `pgvector`, or an external API).

### Adding new audio formats

1. Add the extension to `SUPPORTED_EXTENSIONS` in `mr/config.py`.
2. Ensure `ffmpeg` can decode it (most formats are supported out of the box).

### Using an external API (e.g., MusicBrainz, AcoustID API)

If you want to query AcoustID's online API instead of a local database:
1. Get a free API key from [acoustid.org](https://acoustid.org/).
2. Add it as a GitHub Actions secret (`ACOUSTID_API_KEY`).
3. Replace the FAISS query in `mr/search.py` with an HTTP call to `https://api.acoustid.org/v2/lookup`.

### Scaling beyond GitHub Actions

For very large datasets or high throughput:
- Use a self-hosted runner with more RAM/CPU.
- Replace FAISS with a persistent vector database (Milvus, Pinecone, Weaviate).
- Use PostgreSQL + `pgvector` for combined metadata + vector search.

---

## Testing

### Run tests locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -v
```

### Test structure

| Test file | Covers |
|-----------|--------|
| `test_config.py` | Configuration defaults, directory creation |
| `test_audio.py` | Audio normalisation and fingerprinting (mocked ffmpeg/fpcalc) |
| `test_db.py` | Fingerprint vector conversion |
| `test_search.py` | Distance-to-confidence mapping, search logic |
| `test_models.py` | Data class serialisation / deserialisation |
| `test_summary.py` | GitHub Actions summary table generation |
| `test_utils.py` | File discovery, logging setup |
| `test_integration.py` | End-to-end with a tiny FAISS index |

### CI

The **🧪 CI Tests** workflow runs automatically on every push and PR to `main`, testing across Python 3.10, 3.11, and 3.12.

---

## License & Attribution

### Code

This project's code is available under your chosen license.

### Data

The AcoustID database is provided under the [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license.

> **Attribution:** This project uses data from the [AcoustID](https://acoustid.org/) project by Lukáš Lalinský.

If you redistribute results, include this attribution.
