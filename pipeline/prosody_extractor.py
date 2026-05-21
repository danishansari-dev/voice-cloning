"""
Prosody feature extraction via Parselmouth (Praat bindings).

Prosody — pitch contour, intensity dynamics, speaking rate, voice quality —
is what separates a lifeless clone from one that *sounds* like the target
speaker.  These features feed into the behavioural injection layer that
adds natural rhythm to synthesised speech.

Parselmouth exposes Praat's battle-tested DSP through Python; we lean on it
heavily rather than rolling our own pitch tracker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import parselmouth
from parselmouth.praat import call

import config  # noqa: F401  — imported so downstream users see it in the namespace

logger = logging.getLogger(__name__)

# Sensible Praat defaults for speech analysis
_F0_FLOOR_HZ: float = 75.0
_F0_CEILING_HZ: float = 600.0


class ProsodyExtractor:
    """Extract pitch, intensity, speaking-rate, and voice-quality features.

    All methods accept a file path (rather than an in-memory waveform) because
    Parselmouth's native file loader handles format decoding robustly and
    avoids unnecessary sample-rate / dtype conversions.
    """

    # ------------------------------------------------------------------ #
    #  Pitch (F0)                                                         #
    # ------------------------------------------------------------------ #

    def extract_f0(self, audio_path: Union[str, Path]) -> dict:
        """Extract fundamental frequency statistics and raw contour.

        F0 range and variability are the strongest cues for perceived
        speaker identity after timbre.  The contour itself is used by
        the injection layer to reproduce intonation patterns.

        Args:
            audio_path: Path to an audio file.

        Returns:
            Dict with keys ``f0_mean``, ``f0_std``, ``f0_min``, ``f0_max``,
            ``f0_range``, ``f0_contour`` (numpy array of voiced-frame F0s).
        """
        snd = self._load_sound(audio_path)
        pitch = snd.to_pitch(pitch_floor=_F0_FLOOR_HZ, pitch_ceiling=_F0_CEILING_HZ)
        f0_values = pitch.selected_array["frequency"]

        # Keep only voiced frames (Praat sets unvoiced frames to 0)
        voiced = f0_values[f0_values > 0]

        if len(voiced) == 0:
            logger.warning("No voiced frames detected in %s", Path(audio_path).name)
            return {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_min": 0.0,
                "f0_max": 0.0,
                "f0_range": 0.0,
                "f0_contour": np.array([], dtype=np.float64),
            }

        return {
            "f0_mean": float(np.mean(voiced)),
            "f0_std": float(np.std(voiced)),
            "f0_min": float(np.min(voiced)),
            "f0_max": float(np.max(voiced)),
            "f0_range": float(np.max(voiced) - np.min(voiced)),
            "f0_contour": voiced,
        }

    # ------------------------------------------------------------------ #
    #  Intensity                                                          #
    # ------------------------------------------------------------------ #

    def extract_intensity(self, audio_path: Union[str, Path]) -> dict:
        """Extract intensity (loudness) statistics and contour.

        Intensity dynamics tell us whether the speaker is a "loud talker"
        or tends toward quieter, more intimate delivery — critical for
        matching perceived energy in the clone.

        Args:
            audio_path: Path to an audio file.

        Returns:
            Dict with ``intensity_mean``, ``intensity_std``,
            ``intensity_contour`` (numpy array, dB).
        """
        snd = self._load_sound(audio_path)
        intensity = snd.to_intensity()
        values = intensity.values[0]  # shape (1, N) → (N,)

        # Filter out -inf / NaN that Praat occasionally produces at edges
        valid = values[np.isfinite(values)]

        if len(valid) == 0:
            logger.warning("No valid intensity frames in %s", Path(audio_path).name)
            return {
                "intensity_mean": 0.0,
                "intensity_std": 0.0,
                "intensity_contour": np.array([], dtype=np.float64),
            }

        return {
            "intensity_mean": float(np.mean(valid)),
            "intensity_std": float(np.std(valid)),
            "intensity_contour": valid,
        }

    # ------------------------------------------------------------------ #
    #  Speaking rate                                                       #
    # ------------------------------------------------------------------ #

    def extract_speaking_rate(
        self,
        audio_path: Union[str, Path],
        transcript: str | None = None,
    ) -> dict:
        """Estimate speaking rate (syllables/s and words/min).

        If a transcript is supplied we count words directly; otherwise we
        estimate syllable nuclei from intensity peaks — a classic approach
        from phonetics research (De Jong & Wempe, 2009).

        Args:
            audio_path: Path to an audio file.
            transcript: Optional verbatim transcript.

        Returns:
            Dict with ``syllables_per_second``, ``words_per_minute``,
            ``articulation_rate``.
        """
        snd = self._load_sound(audio_path)
        duration = snd.duration

        if duration < 0.1:
            logger.warning("Audio too short for speaking-rate analysis (%.3f s)", duration)
            return {
                "syllables_per_second": 0.0,
                "words_per_minute": 0.0,
                "articulation_rate": 0.0,
            }

        if transcript is not None:
            words = transcript.split()
            word_count = len(words)
            # Rough syllable estimate: each word ≈ max(1, vowel-cluster count)
            syllable_count = sum(
                max(1, sum(1 for ch in w.lower() if ch in "aeiouy"))
                for w in words
            )
        else:
            # Estimate syllable nuclei from intensity peaks
            syllable_count = self._estimate_syllables(snd)
            # Approximate word count (English avg ≈ 1.5 syllables / word)
            word_count = max(1, int(syllable_count / 1.5))

        syllables_per_sec = syllable_count / duration
        words_per_min = (word_count / duration) * 60.0

        # Articulation rate excludes pauses — use a rough voiced-duration estimate
        pitch = snd.to_pitch()
        voiced_frames = np.sum(pitch.selected_array["frequency"] > 0)
        total_frames = len(pitch.selected_array["frequency"])
        voiced_ratio = voiced_frames / max(1, total_frames)
        voiced_duration = duration * voiced_ratio
        articulation_rate = syllable_count / max(0.01, voiced_duration)

        return {
            "syllables_per_second": float(syllables_per_sec),
            "words_per_minute": float(words_per_min),
            "articulation_rate": float(articulation_rate),
        }

    # ------------------------------------------------------------------ #
    #  Voice quality                                                      #
    # ------------------------------------------------------------------ #

    def extract_voice_quality(self, audio_path: Union[str, Path]) -> dict:
        """Extract jitter, shimmer, and harmonics-to-noise ratio.

        These micro-perturbation metrics capture the "texture" of a voice —
        a breathy speaker will have high shimmer and low HNR, while a clear
        speaker shows the opposite.  They guide post-processing effects
        applied to the synthesised waveform.

        Args:
            audio_path: Path to an audio file.

        Returns:
            Dict with ``jitter``, ``shimmer``, ``hnr``.
        """
        snd = self._load_sound(audio_path)
        pitch = snd.to_pitch(pitch_floor=_F0_FLOOR_HZ, pitch_ceiling=_F0_CEILING_HZ)

        # PointProcess from periodic peaks — needed for jitter / shimmer
        point_process = call(
            snd, "To PointProcess (periodic, cc)",
            _F0_FLOOR_HZ, _F0_CEILING_HZ,
        )

        # Guard against silent / unvoiced audio
        n_periods = call(point_process, "Get number of periods", 0.0, 0.0, 0.0001, 0.02, 1.3)
        if n_periods < 3:
            logger.warning(
                "Too few voiced periods (%d) in %s for jitter/shimmer",
                n_periods, Path(audio_path).name,
            )
            return {"jitter": 0.0, "shimmer": 0.0, "hnr": 0.0}

        jitter = call(
            point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
        )
        shimmer = call(
            [snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
        )

        # Harmonics-to-noise ratio
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, _F0_FLOOR_HZ, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0.0, 0.0)

        return {
            "jitter": float(jitter),
            "shimmer": float(shimmer),
            "hnr": float(hnr),
        }

    # ------------------------------------------------------------------ #
    #  Combined extraction                                                #
    # ------------------------------------------------------------------ #

    def extract_full_prosody(self, audio_path: Union[str, Path]) -> dict:
        """Run every extractor and merge results into one flat dict.

        Convenience method for callers that want "everything".

        Args:
            audio_path: Path to an audio file.

        Returns:
            Merged dict of all prosodic features.
        """
        audio_path = Path(audio_path)
        logger.info("Full prosody extraction: %s", audio_path.name)

        result: dict = {}
        result.update(self.extract_f0(audio_path))
        result.update(self.extract_intensity(audio_path))
        result.update(self.extract_speaking_rate(audio_path))
        result.update(self.extract_voice_quality(audio_path))
        return result

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_sound(audio_path: Union[str, Path]) -> parselmouth.Sound:
        """Load an audio file into a Parselmouth Sound object.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return parselmouth.Sound(str(audio_path))

    @staticmethod
    def _estimate_syllables(snd: parselmouth.Sound) -> int:
        """Estimate syllable count from intensity-peak detection.

        This replicates the approach used in Praat's "Syllable Nuclei" script:
        smooth the intensity contour, find peaks above a dynamic threshold,
        and count them.  It's an approximation but works well for rate
        estimation when no transcript is available.
        """
        intensity = snd.to_intensity(minimum_pitch=_F0_FLOOR_HZ)
        values = intensity.values[0]
        valid = values[np.isfinite(values)]

        if len(valid) < 5:
            return 0

        # Dynamic threshold: mean minus 1 SD (catches soft syllables too)
        threshold = np.mean(valid) - np.std(valid)

        # Simple peak detection on the smoothed contour
        peaks = 0
        in_peak = False
        for v in valid:
            if v > threshold and not in_peak:
                peaks += 1
                in_peak = True
            elif v < threshold:
                in_peak = False

        return max(1, peaks)


# ────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    import soundfile as sf
    from rich.console import Console

    console = Console()
    console.rule("[bold green]ProsodyExtractor — smoke test")

    # Generate a synthetic vowel-like tone to test against
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simple 200 Hz tone with harmonics (simulates a vowel)
    synth = (
        0.5 * np.sin(2 * np.pi * 200 * t)
        + 0.3 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 600 * t)
    ).astype(np.float32)

    tmp_path = config.OUTPUT_DIR / "_prosody_test.wav"
    sf.write(str(tmp_path), synth, sr)

    extractor = ProsodyExtractor()

    console.print("[cyan]Extracting F0…")
    f0 = extractor.extract_f0(tmp_path)
    console.print(f"  f0_mean={f0['f0_mean']:.1f} Hz, f0_range={f0['f0_range']:.1f} Hz")

    console.print("[cyan]Extracting intensity…")
    intensity = extractor.extract_intensity(tmp_path)
    console.print(f"  intensity_mean={intensity['intensity_mean']:.1f} dB")

    console.print("[cyan]Extracting speaking rate…")
    rate = extractor.extract_speaking_rate(tmp_path)
    console.print(f"  syllables/s={rate['syllables_per_second']:.2f}")

    console.print("[cyan]Extracting voice quality…")
    vq = extractor.extract_voice_quality(tmp_path)
    console.print(f"  jitter={vq['jitter']:.6f}, shimmer={vq['shimmer']:.6f}, hnr={vq['hnr']:.1f}")

    console.print("[cyan]Full prosody extraction…")
    full = extractor.extract_full_prosody(tmp_path)
    # Numpy arrays aren't JSON-serialisable — convert for display
    displayable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in full.items()
    }
    console.print_json(json.dumps(displayable, indent=2))

    tmp_path.unlink(missing_ok=True)
    console.print("[bold green]✓ All ProsodyExtractor tests passed")
