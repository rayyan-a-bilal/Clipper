# Clipper Pipeline Specification

## Overview
This document outlines the complete video clipping pipeline for producing high-quality vertical short clips from long input videos.

## Pipeline Stages

1. **ingest**: 
   - Watch folder / CLI / web upload
   - Idempotent file hash
   - Store in data directory

2. **probe/proxy**:
   - ffprobe metadata
   - 960x540@15fps proxy video
   - 16kHz mono WAV audio 

3. **transcribe**:
   - faster-whisper large-v3-turbo on GPU0
   - word_timestamps=True, vad_filter=True, condition_on_previous_text=False

4. **analyze**:
   - PySceneDetect shots
   - ffmpeg silencedetect -35dB/0.5s
   - RMS energy in 0.5s windows 
   - YuNet face detection on CPU at 3fps
   - IoU face tracking

5. **candidates**:
   - 3-8 sentence windows, 15-75s
   - Cheap heuristic score
   - Keep top 20

6. **LLM scoring**:
   - Rubric axes: hook/payoff/self_contained/emotion/clarity (0-10 each)  
   - JSON-schema constrained
   - Batches of 5
   - Retry twice

7. **selection/quality gate**:
   - Weighted score = 0.30*hook + 0.25*payoff + 0.20*self_contained + 0.15*emotion + 0.10*clarity
   - Gate: weighted >= 7.2, hook >= 6, self_contained >= 7
   - Target 3 clips, max 5
   - Dedupe overlap > 30%
   - Zero clips is a valid successful result

8. **cut refinement**:
   - LLM only proposes in/out 
   - Code computes final cuts from word/sentence boundaries
   - Padding, silence snapping <= 300ms, shot boundary snapping <= 400ms

9. **reframe render**:
   - Python/cv2 frameserver from ORIGINAL video to 1080x1920 mezzanine
   - CRF 14

10. **captions/compose**:
    - ffmpeg ASS captions + loudnorm I=-16:TP=-1.5:LRA=11 + h264_nvenc cq19, with libx264 fallback
    - Words -> ASS, <= 4 words/group, <= 20 chars/line, max 2 lines
    - \kf karaoke highlight yellow -> white, Montserrat ExtraBold
    - Size ~88, MarginV 450, safe zones (right 150px / top 220px)
    - Hook overlay top, first 2.5s only

11. **QC review**:
    - FastAPI + Jinja2 + HTMX
    - Preview, approve/reject, layout override, in/out nudge, caption edit, re-render, export

12. **export**:
    - Final vertical clips for social platforms

13. **future posting outbox**:
    - Outbox table + PostAdapter stub only in v1

## Data Flow
Every future stage: typed input -> JSON artifact with schema_version=1 -> typed output.

Stages must be resumable: skip valid completed stages.