# Clipper Project Rules

## Development Environment
- Python 3.11
- pydantic v2
- typer
- ruff
- pytest

## Architecture
- All data contracts live ONLY in `src/clipper/models/`
- All GPU/ffmpeg/network/model access must go through `adapters/`
- Every adapter must have a CPU-only Fixture/Mock for tests
- Pure logic must be separated from I/O
- Tests must never require GPU, CUDA, live LLM, model downloads, or root

## Constraints
- No new dependencies without asking first
- No sudo/systemd/model downloads
- One milestone at a time only
- Blocked after 3 attempts -> write BLOCKED.md and ask
- Definition of done: acceptance commands pass, ruff clean, tests green

## Future Milestones (M0-M10)
M0 - Scaffold & CLI
M1 - Input processing pipeline
M2 - Transcription
M3 - Scene analysis
M4 - Clip candidates generation
M5 - LLM scoring
M6 - Quality gate & selection
M7 - Cut refinement
M8 - Reframe rendering
M9 - Caption generation/composition
M10 - QC workflow + Outbox

## Testing Rules
- All tests must run in CPU-only environment
- No GPU/CUDA dependencies
- No live LLM calls
- No model downloads
- No root privileges required
