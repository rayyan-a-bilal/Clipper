# Clipper - BLOCKED Items

This file is created to document items that were blocked during M0 development, but are not currently required for completion. 

## Currently Blocked Items (None)

All requirements in the M0 specification have been successfully implemented and the project is ready for the next milestone.

The implementation includes:
✅ Project scaffolding with proper directory structure
✅ All core data contracts  
✅ Configuration system with environment variable support
✅ SQLite database with schema migrations
✅ Device detection (graceful fallbacks) 
✅ CLI with doctor command only
✅ Complete documentation files
✅ Test fixtures and basic testing infrastructure

## Items Not Implemented in M0 

These will be implemented in future milestones:

1. Input processing pipeline (M1)
2. Transcription system (M2)
3. Scene analysis (M3)  
4. Clip candidate generation (M4)
5. LLM scoring system (M5)
6. Quality gate and selection (M6)
7. Cut refinement (M7)
8. Reframe rendering (M8)
9. Caption generation (M9)
10. QC workflow with web UI (M10)
11. Future posting adapter (M10)

## Next Milestone

M0, M1, and M2 (including M2a/M2b) are complete. The last fix in M2 corrected the sentence builder's gap-based splitting (tests/test_sentences.py now passes 6/6). The project is ready for M3 development (Scene analysis).