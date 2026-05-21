"""
Speech-emotion recognition via Wav2Vec2.

Emotion shapes *how* something is said even more than prosody does.
This module classifies 3-second windows into seven emotion categories and
aggregates them into an "emotion fingerprint" that captures the speaker's
typical affective range — e.g., a podcast host who's mostly neutral with
occasional excitement, vs. a storyteller with wide emotional swings.

Primary model : ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
Fallback model: superb/wav2vec2-base-superb-er
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

import config

logger = logging.getLogger(__name__)

# Both emotion models expect 16 kHz input
_EMOTION_SR: int = 16_000

# Default emotion labels for the primary model
_EMOTION_LABELS: list[str] = [
    "angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"
]

# Valence / arousal mappings for fingerprint derivation
# Values are rough dimensional-emotion coordinates (Russell's circumplex)
_VALENCE_MAP: dict[str, float] = {
    "happy": 0.8, "surprised": 0.5, "neutral": 0.0,
    "sad": -0.6, "fearful": -0.4, "angry": -0.3, "disgusted": -0.5,
}
_AROUSAL_MAP: dict[str, float] = {
    "angry": 0.8, "fearful": 0.7, "surprised": 0.7, "happy": 0.6,
    "disgusted": 0.4, "neutral": 0.1, "sad": -0.3,
}


class EmotionAnalyzer:
    """Classify speech emotion per segment and build an emotion fingerprint.

    The model is loaded **once** at construction time and kept on
    ``config.DEVICE`` to avoid repeated cold starts.
    """

    def __init__(self) -> None:
        """Load the primary emotion model; fall back if it fails."""
        self._device = config.DEVICE
        self._labels: list[str] = _EMOTION_LABELS
        self._model, self._feature_extractor = self._load_model(
            config.EMOTION_MODEL_PRIMARY
        )
        if self._model is None:
            logger.warning("Primary model failed — trying fallback")
            self._model, self._feature_extractor = self._load_model(
                config.EMOTION_MODEL_FALLBACK
            )
        if self._model is None:
            raise RuntimeError("Could not load any emotion recognition model")

    def _load_model(
        self, model_name: str
    ) -> tuple[Wav2Vec2ForSequenceClassification | None, AutoFeatureExtractor | None]:
        """Attempt to load a HuggingFace emotion model.

        Returns ``(None, None)`` instead of raising so the caller can try
        the fallback model without extra exception handling.
        """
        try:
            logger.info("Loading emotion model: %s …", model_name)
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            model = model.to(self._device)
            model.eval()

            # Update labels from model config if available
            if hasattr(model.config, "id2label") and model.config.id2label:
                self._labels = [
                    model.config.id2label[i]
                    for i in range(len(model.config.id2label))
                ]

            logger.info("Emotion model ready (%d labels).", len(self._labels))
            return model, feature_extractor
        except Exception as exc:
            logger.error("Failed to load %s: %s", model_name, exc)
            return None, None

    # ------------------------------------------------------------------ #
    #  Single-segment analysis                                            #
    # ------------------------------------------------------------------ #

    def analyze_segment(
        self, audio_segment: np.ndarray, sr: int
    ) -> dict:
        """Classify emotion in a short audio segment.

        Args:
            audio_segment: 1-D float32 waveform (ideally ≤ 5 s).
            sr: Sample rate.

        Returns:
            Dict with ``emotion`` (top label), ``confidence`` (0-1),
            and ``all_scores`` mapping each label to its softmax score.
        """
        # Resample to 16 kHz if needed
        if sr != _EMOTION_SR:
            audio_segment = librosa.resample(
                audio_segment, orig_sr=sr, target_sr=_EMOTION_SR
            )

        # Feature extraction + inference
        inputs = self._feature_extractor(
            audio_segment,
            sampling_rate=_EMOTION_SR,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self._device)

        with torch.no_grad():
            logits = self._model(input_values).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Build label → score map
        all_scores: dict[str, float] = {}
        for i, label in enumerate(self._labels):
            score = float(probs[i]) if i < len(probs) else 0.0
            all_scores[label] = score

        top_idx = int(np.argmax(probs))
        top_label = self._labels[top_idx] if top_idx < len(self._labels) else "unknown"
        top_confidence = float(probs[top_idx])

        return {
            "emotion": top_label,
            "confidence": top_confidence,
            "all_scores": all_scores,
        }

    # ------------------------------------------------------------------ #
    #  Full-audio analysis (sliding window)                               #
    # ------------------------------------------------------------------ #

    def analyze_full_audio(
        self, audio_path: Union[str, Path]
    ) -> list[dict]:
        """Classify emotion for every 3-second window in a file.

        The 3-second window size balances temporal resolution with giving
        the model enough context to make a meaningful prediction.

        Args:
            audio_path: Path to an audio file.

        Returns:
            List of dicts, each containing ``start``, ``end``,
            ``emotion``, ``confidence``, ``all_scores``.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sr = librosa.load(str(audio_path), sr=_EMOTION_SR, mono=True)
        total_duration = len(waveform) / sr

        window_sec = 3.0
        hop_sec = 3.0  # non-overlapping windows
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        results: list[dict] = []
        for start_sample in range(0, len(waveform) - window_samples + 1, hop_samples):
            segment = waveform[start_sample : start_sample + window_samples]
            start_t = start_sample / sr
            end_t = (start_sample + window_samples) / sr

            analysis = self.analyze_segment(segment, sr)
            analysis["start"] = float(start_t)
            analysis["end"] = float(end_t)
            results.append(analysis)

        # Handle the tail if it's at least 1 s long
        remaining_start = (len(results) * hop_samples)
        remaining = waveform[remaining_start:]
        if len(remaining) >= sr:
            analysis = self.analyze_segment(remaining, sr)
            analysis["start"] = float(remaining_start / sr)
            analysis["end"] = float(total_duration)
            results.append(analysis)

        logger.info("Analysed %d emotion windows in %s", len(results), audio_path.name)
        return results

    # ------------------------------------------------------------------ #
    #  Emotion fingerprint                                                #
    # ------------------------------------------------------------------ #

    def build_emotion_fingerprint(
        self, audio_list: list[tuple[np.ndarray, int]]
    ) -> dict:
        """Aggregate emotion across multiple clips into a speaker fingerprint.

        The fingerprint captures the speaker's *typical* emotional profile:
        which emotions dominate, how wide their emotional range is, and
        their baseline arousal/valence.

        Args:
            audio_list: List of ``(waveform, sr)`` pairs.

        Returns:
            Dict with ``dominant_emotion``, ``emotion_distribution``,
            ``emotional_range``, ``baseline_arousal``, ``baseline_valence``.
        """
        if not audio_list:
            raise ValueError("audio_list is empty")

        # Analyse every clip in 3 s windows
        all_results: list[dict] = []
        for waveform, sr in audio_list:
            # Resample once for consistency
            if sr != _EMOTION_SR:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=_EMOTION_SR)
                sr = _EMOTION_SR

            window = int(3.0 * sr)
            hop = window
            for i in range(0, len(waveform) - window + 1, hop):
                seg = waveform[i : i + window]
                all_results.append(self.analyze_segment(seg, sr))

            # Tail
            tail = waveform[(len(waveform) // window) * window :]
            if len(tail) >= sr:
                all_results.append(self.analyze_segment(tail, sr))

        if not all_results:
            logger.warning("No segments long enough for emotion analysis")
            return self._empty_fingerprint()

        # --- Emotion distribution (% time in each state) ----------------
        counts: dict[str, int] = {label: 0 for label in self._labels}
        for r in all_results:
            em = r["emotion"]
            if em in counts:
                counts[em] += 1

        total = sum(counts.values()) or 1
        distribution = {label: round(c / total, 4) for label, c in counts.items()}

        # --- Dominant emotion -------------------------------------------
        dominant = max(distribution, key=distribution.get)  # type: ignore[arg-type]

        # --- Emotional range: entropy of distribution (0 = one emotion only) -
        probs = np.array(list(distribution.values()))
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        max_entropy = np.log2(len(self._labels))
        emotional_range = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        # --- Baseline arousal / valence ---------------------------------
        arousals = [_AROUSAL_MAP.get(r["emotion"], 0.0) for r in all_results]
        valences = [_VALENCE_MAP.get(r["emotion"], 0.0) for r in all_results]
        # Normalise to [0, 1]
        baseline_arousal = float((np.mean(arousals) + 1) / 2)
        baseline_valence = float((np.mean(valences) + 1) / 2)

        return {
            "dominant_emotion": dominant,
            "emotion_distribution": distribution,
            "emotional_range": round(emotional_range, 4),
            "baseline_arousal": round(baseline_arousal, 4),
            "baseline_valence": round(baseline_valence, 4),
        }

    def _empty_fingerprint(self) -> dict:
        """Zeroed-out fingerprint for edge cases."""
        return {
            "dominant_emotion": "neutral",
            "emotion_distribution": {label: 0.0 for label in self._labels},
            "emotional_range": 0.0,
            "baseline_arousal": 0.5,
            "baseline_valence": 0.5,
        }


# ────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    console.rule("[bold green]EmotionAnalyzer — smoke test")

    analyzer = EmotionAnalyzer()

    sr = 16_000
    duration = 5.0
    np.random.seed(42)
    # Synthetic noise — not real speech but exercises the full code path
    fake_audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.05

    console.print("[cyan]Analysing a single segment…")
    result = analyzer.analyze_segment(fake_audio, sr)
    console.print(f"  Emotion: {result['emotion']} ({result['confidence']:.2%})")
    console.print(f"  All scores: {result['all_scores']}")

    console.print("[cyan]Building emotion fingerprint from 2 clips…")
    clips = [(fake_audio, sr), (fake_audio * 1.5, sr)]
    fp = analyzer.build_emotion_fingerprint(clips)
    console.print(f"  Dominant: {fp['dominant_emotion']}")
    console.print(f"  Distribution: {fp['emotion_distribution']}")
    console.print(f"  Range: {fp['emotional_range']:.3f}")
    console.print(f"  Arousal: {fp['baseline_arousal']:.3f}, Valence: {fp['baseline_valence']:.3f}")

    console.print("[bold green]✓ All EmotionAnalyzer tests passed")
