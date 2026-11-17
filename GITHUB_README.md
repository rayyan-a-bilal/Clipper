# Clipper - Local, Headless Video Clipping Pipeline

Clipper is a local, headless video clipping pipeline designed for content creation workflows. It processes video files through multiple stages including probing, proxy generation, audio extraction, and candidate analysis.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with clearly defined pipeline stages
- **Headless Processing**: Can be run without a GUI, perfect for server environments
- **Database Integration**: SQLite-based job tracking and state management
- **Device Detection**: Automatic detection of available hardware (NVIDIA GPUs, FFmpeg)
- **Extensible**: Easy to add new processing stages or modify existing ones

## 📋 Pipeline Stages

1. **Ingest**: File watching, content hashing, initial job creation
2. **Probe/Proxy**: Metadata extraction and proxy video generation  
3. **Transcribe**: Audio transcription with word timestamps using faster-whisper
4. **Analyze**: Scene detection, silence detection, face tracking
5. **Candidates**: Generate candidate clip windows using heuristic scoring
6. **LLM Scoring**: LLM-based quality scoring of candidate clips
7. **Selection/Quality Gate**: Final clip selection based on weighted scores
8. **Cut Refinement**: Fine-tuning cut timing and boundaries
9. **Reframe Rendering**: Final vertical clip rendering (1080x1920)
10. **Caption Generation**: Subtitle generation and positioning

## 🛠️ Requirements

- Python 3.11+
- FFmpeg (for media processing)  
- NVIDIA GPU (optional, for accelerated processing)

## 📦 Installation

```bash
# Clone or download the repository
git clone <repository-url>
cd clipper

# Create a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e .
```

## 💻 Usage

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

## 📁 Project Structure

```
clipper/
├── src/clipper/                 # Core application code
│   ├── cli.py                   # Command line interface
│   ├── config.py                # Configuration management  
│   ├── db.py                    # Database operations
│   ├── devices.py               # Device detection and management
│   ├── stages/                  # Pipeline stage implementations
│   │   └── ingest.py            # Ingest pipeline stage
│   ├── adapters/                # Media processing adapters
│   │   └── media.py             # Media processing interface
│   ├── models/                  # Data models and contracts
│   ├── web/                     # Web components (planned)
│   └── worker.py               # Worker implementation (planned)
├── tests/                       # Test suite
├── scripts/                     # Helper scripts
│   └── make_fixtures.py         # Test fixture creation script
├── docs/                        # Documentation
│   └── spec/                    # Specification documents
├── data/                        # Data directory (created by app)
└── README.md
```

## 🧪 Testing

```bash
# Run tests for all components
python -m pytest tests/

# Run tests excluding GPU tests
python -m pytest -m "not gpu"

# Run ruff linter
python -m ruff check .
```

## 📚 Documentation

- `docs/spec/pipeline.md` - Complete pipeline specification
- `docs/spec/milestones.md` - Milestone development plan  
- `docs/spec/contracts.md` - Data contract definitions
- `docs/spec/scoring_prompt.md` - LLM scoring prompt template

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

MIT License