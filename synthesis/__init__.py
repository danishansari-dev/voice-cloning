"""
Synthesis package for the AI Voice Cloning System.

Centralises exports so consumers can do:
    from synthesis import TextProcessor, TTSEngine, AudioStitcher, ...
instead of reaching into individual submodules.
"""

from synthesis.text_processor import TextProcessor, Sentence, SynthSegment
from synthesis.bark_events import BarkEventGenerator
from synthesis.behavioral_injector import BehavioralInjector
from synthesis.emotion_conditioner import EmotionConditioner
from synthesis.tts_engine import TTSEngine
from synthesis.audio_stitcher import AudioStitcher

__all__ = [
    "TextProcessor",
    "Sentence",
    "SynthSegment",
    "BarkEventGenerator",
    "BehavioralInjector",
    "EmotionConditioner",
    "TTSEngine",
    "AudioStitcher",
]
