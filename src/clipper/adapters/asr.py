"""
ASR Adapter protocol and implementations for clipper.
"""

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..models import Word


class ASRAdapter(ABC):
    """Abstract base class for ASR adapters."""

    @abstractmethod
    def transcribe(self, wav_path: Path) -> list[Word]:
        """Transcribe audio file to words."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close any resources used by the adapter."""
        pass


class FixtureASR(ASRAdapter):
    """Fixture ASR that loads from JSON for testing."""

    def __init__(self, fixture_path: Path = None):
        """
        Initialize the fixture ASR.
        
        Args:
            fixture_path: Path to JSON fixture file containing words data
        """
        self.fixture_path = fixture_path or Path("tests/fixtures/words.json")
        self._loaded_words = None

    def transcribe(self, wav_path: Path) -> list[Word]:
        """Load words from fixture file."""
        import json

        with open(self.fixture_path) as f:
            data = json.load(f)

        # Handle both list format and dict format
        if isinstance(data, list):
            words_data = data
        elif isinstance(data, dict) and "words" in data:
            words_data = data["words"]
        else:
            raise ValueError("Fixture must contain either a list of words or a dict with 'words' key")

        # Validate and create Word objects (skip empty text words, like real ASR adapter)
        words = []
        for word_data in words_data:
            text = (word_data or {}).get("text", "")
            if not text.strip():
                continue
            word = Word(**word_data)
            words.append(word)

        self._loaded_words = words
        return words

    def close(self) -> None:
        """No-op close method."""
        pass


class FasterWhisperASR(ASRAdapter):
    """Faster whisper ASR adapter that loads model lazily on first use."""

    def __init__(self, cfg):
        """
        Initialize the faster-whisper ASR adapter.
        
        Args:
            cfg: Configuration object containing device and model settings
        """
        self.cfg = cfg
        self._model = None
        self._model_device = None

    def _load_model(self):
        """Load the whisper model lazily."""
        if self._model is not None:
            return  # Already loaded
            
        # Import faster_whisper only when needed to avoid CPU-only test dependency issues
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError("faster_whisper not installed. Please install with: pip install faster-whisper") from e
            
        # Parse device from config 
        device = self.cfg.asr.device if hasattr(self.cfg.asr, 'device') else "cuda:0"
        
        # Handle device parsing and configuration
        if device == "cpu":
            self._model_device = "cpu"
            compute_type = "float32"
        else:
            # For cuda devices like "cuda:0", we'll use gpu
            # Parse GPU ID from device string
            try:
                device_index = int(device.split(":")[1]) if ":" in device else 0
            except (ValueError, IndexError):
                device_index = 0
            
            self._model_device = "cuda"
            # Extract compute type from config or default to float16
            compute_type = self.cfg.asr.compute_type if hasattr(self.cfg.asr, 'compute_type') else "float16"
        
        # Load model with the specified parameters
        try:
            # Using device_index approach as recommended by faster_whisper docs
            self._model = WhisperModel(
                model_size_or_path=self.cfg.asr.model or "large-v3-turbo",
                device=self._model_device,
                compute_type=compute_type,
                device_index=device_index if self._model_device == "cuda" else None
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e

    def transcribe(self, wav_path: Path) -> List[Word]:
        """Transcribe audio file using faster-whisper."""
        if self._model is None:
            self._load_model()
            
        # Run transcription with specified parameters
        segments, info = self._model.transcribe(
            str(wav_path),
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            beam_size=5
        )
        
        # Convert to Word objects (as per task requirements)
        words = []
        for segment in segments:
            for word in segment.words:
                # Skip empty text words
                if not word.word.strip():
                    continue
                    
                word_obj = Word(
                    start=float(word.start),
                    end=float(word.end),
                    text=word.word.strip(),
                    prob=float(word.probability)
                )
                words.append(word_obj)
                
        return words

    def close(self) -> None:
        """Close the model and free resources."""
        if self._model is not None:
            # Clear the model reference
            self._model = None
            # Force garbage collection to release memory
            gc.collect()


def get_adapter(cfg):
    """
    Factory function to get appropriate ASR adapter.
    
    Args:
        cfg: Configuration object
        
    Returns:
        ASRAdapter instance 
    """
    return FasterWhisperASR(cfg)