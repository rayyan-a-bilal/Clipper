# Milestone Specifications

## M0 - Scaffold & CLI (Current)
- Project scaffolding with all directory structure
- Basic config system using pydantic-settings
- SQLite DB with basic migrations
- Core data models (contracts)
- CLI with doctor command only
- Device detection and locking
- Testing framework setup

## M1 - Input processing pipeline
- File watching/processing
- Proxy video generation
- Audio extraction 
- Metadata extraction
- Job state management

## M2 - Transcription
- ASR integration (faster-whisper)
- Word timestamps
- VAD filtering
- Audio quality checks

## M3 - Scene analysis  
- PySceneDetect shot detection
- Silence detection
- RMS energy analysis
- Face detection on CPU
- Face tracking across frames

## M4 - Clip candidates generation
- Candidate window generation
- Heuristic scoring
- Candidate storage and retrieval
- Window filtering

## M5 - LLM scoring
- LLM integration (OpenAI-compatible)
- Batch processing 
- Retry logic
- Scoring result storage

## M6 - Quality gate & selection
- Weighted scoring system
- Quality thresholds
- Duplicate detection
- Final clip selection

## M7 - Cut refinement
- In/out timing optimization
- Word boundary snapping
- Silence and shot boundary snapping
- Cut validation 

## M8 - Reframe rendering 
- Original video frame server
- Frame resolution conversion
- CRF 14 encoding

## M9 - Caption generation/composition
- ASS subtitle generation  
- Karaoke effects
- Font handling
- Caption positioning
- Loudnorm audio processing

## M10 - QC workflow + Outbox
- Web UI with FastAPI/Jinja2/HTMX
- Preview functionality
- Approval/rejection workflow
- Layout override capability
- Export system 
- Future posting adapter