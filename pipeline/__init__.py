"""
Pipeline package for the AI Voice Cloning System.

Re-exports every pipeline class so callers can do:
    from pipeline import AudioLoader, SpeakerEncoder, ...
instead of reaching into submodules.
"""

from pipeline.audio_loader import AudioLoader
from pipeline.speaker_encoder import SpeakerEncoder
from pipeline.prosody_extractor import ProsodyExtractor
from pipeline.behavioral_profiler import BehavioralProfiler
from pipeline.emotion_analyzer import EmotionAnalyzer
from pipeline.profile_builder import ProfileBuilder, BehavioralProfile

__all__ = [
    "AudioLoader",
    "SpeakerEncoder",
    "ProsodyExtractor",
    "BehavioralProfiler",
    "EmotionAnalyzer",
    "ProfileBuilder",
    "BehavioralProfile",
]
