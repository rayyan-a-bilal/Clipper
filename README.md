# Clipper - Local, Headless Video Clipping Pipeline

Clipper is a local, headless video clipping pipeline designed for content creation workflows. It processes video files through multiple stages including probing, proxy generation, audio extraction, and candidate analysis.

## Requirements

- Python 3.11+
- FFmpeg (for media processing)
- NVIDIA GPU (optional, for accelerated processing)

## Installation

```bash
# Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e .
```

## Usage

### Running Tests
```bash
# Run all tests (excluding GPU tests)
python -m pytest -m "not gpu"

# Run ruff checks
python -m ruff check .
```

### Ingest a Video
```bash
python -m clipper ingest --source-path <path-to-video> --client <client-name> --niche <niche-topic>
```

## Project Structure

- `src/clipper/` - Core application code
  - `cli.py` - Command line interface
  - `config.py` - Configuration management
  - `db.py` - Database operations  
  - `devices.py` - Device detection and management
  - `stages/` - Pipeline stage implementations
  - `adapters/` - Media processing adapters
  - `models/` - Data models

- `tests/` - Test suite
- `scripts/make_fixtures.py` - Script to create test fixtures

## Development

To run all development checks:
```bash
python -m pytest -m "not gpu"
python -m ruff check .
```

## License

MIT License