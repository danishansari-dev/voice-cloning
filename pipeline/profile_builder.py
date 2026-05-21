"""
Unified profile builder — orchestrates every pipeline stage.

This is the top-level entry point for voice profiling.  Point it at a
directory of audio files and it produces a ``BehavioralProfile`` that
captures everything the synthesis layer needs to reproduce the speaker's
voice *and* their speaking habits (pauses, breaths, fillers, emotion).

Design choice: we save intermediate artefacts (reference clip, embedding)
to disk so they can be reused without re-running the expensive encoding
step.  Profiles are stored as JSON with numpy arrays converted to lists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

import config
from pipeline.audio_loader import AudioLoader
from pipeline.behavioral_profiler import BehavioralProfiler
from pipeline.emotion_analyzer import EmotionAnalyzer
from pipeline.prosody_extractor import ProsodyExtractor
from pipeline.speaker_encoder import SpeakerEncoder

logger = logging.getLogger(__name__)
console = Console()


# ════════════════════════════════════════════════════════════════════════
#  BehavioralProfile dataclass
# ════════════════════════════════════════════════════════════════════════


@dataclass
class BehavioralProfile:
    """Complete behavioural + acoustic profile of a speaker.

    This is the artefact consumed by the synthesis / injection layer.
    Every field is JSON-serialisable via ``to_dict()`` so profiles can be
    saved, versioned, and compared across recording sessions.
    """

    speaker_id: str
    speaker_embedding: np.ndarray         # shape (192,)
    reference_audio_path: str             # best 6 s clip for XTTS
    prosody: dict                         # pitch, intensity, rate, quality
    pause_profile: dict                   # pause timing statistics
    breathing_profile: dict               # breath events & cadence
    filler_profile: dict                  # filler word habits
    emotion_fingerprint: dict             # dominant emotion, distribution
    whisper_threshold: float              # energy threshold for whisper
    voice_quality: dict                   # jitter, shimmer, HNR
    language: str = "en"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------ #
    #  Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for ``json.dump``.

        Numpy arrays are recursively converted to Python lists so the
        entire structure is JSON-safe.
        """
        d = asdict(self)
        return self._numpy_to_list(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BehavioralProfile:
        """Reconstruct a ``BehavioralProfile`` from a plain dict.

        The ``speaker_embedding`` field is converted back to a numpy array;
        all other fields stay as plain Python types.
        """
        d = dict(d)  # shallow copy to avoid mutating the caller's data
        if "speaker_embedding" in d:
            d["speaker_embedding"] = np.array(d["speaker_embedding"], dtype=np.float64)
        return cls(**d)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_to_list(obj: Any) -> Any:
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: BehavioralProfile._numpy_to_list(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [BehavioralProfile._numpy_to_list(v) for v in obj]
        return obj


# ════════════════════════════════════════════════════════════════════════
#  ProfileBuilder
# ════════════════════════════════════════════════════════════════════════


class ProfileBuilder:
    """Orchestrate all pipeline modules to produce a ``BehavioralProfile``.

    Sub-modules are instantiated lazily on first use so that importing
    ``pipeline`` doesn't trigger heavy model downloads.
    """

    def __init__(self) -> None:
        # Lazy-init — models are only loaded when build_from_directory runs
        self._audio_loader: AudioLoader | None = None
        self._speaker_encoder: SpeakerEncoder | None = None
        self._prosody_extractor: ProsodyExtractor | None = None
        self._behavioral_profiler: BehavioralProfiler | None = None
        self._emotion_analyzer: EmotionAnalyzer | None = None

    # ------------------------------------------------------------------ #
    #  Lazy component access                                              #
    # ------------------------------------------------------------------ #

    def _get_audio_loader(self) -> AudioLoader:
        if self._audio_loader is None:
            self._audio_loader = AudioLoader()
        return self._audio_loader

    def _get_speaker_encoder(self) -> SpeakerEncoder:
        if self._speaker_encoder is None:
            self._speaker_encoder = SpeakerEncoder()
        return self._speaker_encoder

    def _get_prosody_extractor(self) -> ProsodyExtractor:
        if self._prosody_extractor is None:
            self._prosody_extractor = ProsodyExtractor()
        return self._prosody_extractor

    def _get_behavioral_profiler(self) -> BehavioralProfiler:
        if self._behavioral_profiler is None:
            self._behavioral_profiler = BehavioralProfiler()
        return self._behavioral_profiler

    def _get_emotion_analyzer(self) -> EmotionAnalyzer:
        if self._emotion_analyzer is None:
            self._emotion_analyzer = EmotionAnalyzer()
        return self._emotion_analyzer

    # ------------------------------------------------------------------ #
    #  Main build pipeline                                                #
    # ------------------------------------------------------------------ #

    def build_from_directory(
        self,
        audio_dir: Union[str, Path],
        speaker_id: str,
        language: str = "en",
    ) -> BehavioralProfile:
        """Run the full profiling pipeline on a directory of audio files.

        Steps:
        1. Load & preprocess all audio clips.
        2. Select the cleanest 6 s reference clip → save to profiles dir.
        3. Compute averaged speaker embedding.
        4. Extract prosody (F0, intensity, rate, quality) per clip → average.
        5. Profile behavioural habits (pauses, breathing, fillers).
        6. Build emotion fingerprint.
        7. Pack everything into a ``BehavioralProfile``.

        Args:
            audio_dir: Directory containing audio files.
            speaker_id: Unique identifier for this speaker.
            language: ISO 639-1 code (default ``"en"``).

        Returns:
            A fully populated ``BehavioralProfile``.
        """
        audio_dir = Path(audio_dir)
        logger.info("Building profile for speaker '%s' from %s", speaker_id, audio_dir)

        loader = self._get_audio_loader()
        encoder = self._get_speaker_encoder()
        prosody = self._get_prosody_extractor()
        behavior = self._get_behavioral_profiler()
        emotion = self._get_emotion_analyzer()

        # Prepare output directory for this speaker
        speaker_dir = config.PROFILES_DIR / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # ── Step 1: Load audio ──────────────────────────────────────
            task = progress.add_task("Loading audio files …", total=None)
            raw_audio = loader.load_directory(audio_dir)
            progress.update(task, completed=1, total=1)

            # Preprocess all clips
            task = progress.add_task("Preprocessing audio …", total=len(raw_audio))
            processed: list[tuple[np.ndarray, int]] = []
            for wav, sr in raw_audio:
                processed.append(loader.preprocess(wav, sr))
                progress.advance(task)

            # ── Step 2: Reference clip ──────────────────────────────────
            task = progress.add_task("Selecting best reference clip …", total=None)
            ref_clip, ref_sr = loader.get_best_reference_clip(
                processed, target_duration=config.REFERENCE_CLIP_DURATION
            )
            ref_path = loader.save_audio(ref_clip, ref_sr, speaker_dir / "reference.wav")
            progress.update(task, completed=1, total=1)

            # ── Step 3: Speaker embedding ───────────────────────────────
            task = progress.add_task("Computing speaker embedding …", total=None)
            embedding = encoder.encode_multiple(processed)
            encoder.save_embedding(embedding, speaker_dir / "embedding.npy")
            progress.update(task, completed=1, total=1)

            # ── Step 4: Prosody extraction ──────────────────────────────
            # We need on-disk files for Parselmouth, so save temp clips
            task = progress.add_task("Extracting prosody …", total=len(processed))
            prosody_results: list[dict] = []
            tmp_clips: list[Path] = []
            for idx, (wav, sr) in enumerate(processed):
                tmp_path = speaker_dir / f"_tmp_clip_{idx}.wav"
                loader.save_audio(wav, sr, tmp_path)
                tmp_clips.append(tmp_path)
                prosody_results.append(prosody.extract_full_prosody(tmp_path))
                progress.advance(task)

            averaged_prosody = self._average_dicts(prosody_results)

            # ── Step 5: Behavioural profiling ───────────────────────────
            task = progress.add_task("Profiling speech habits …", total=len(tmp_clips))
            pause_results: list[dict] = []
            breath_results: list[dict] = []
            filler_results: list[dict] = []
            whisper_segments_all: list[list[dict]] = []

            for clip_path in tmp_clips:
                pause_results.append(behavior.detect_pauses(clip_path))
                breath_results.append(behavior.detect_breathing(clip_path))
                filler_results.append(behavior.detect_fillers(clip_path))
                whisper_segments_all.append(behavior.detect_whisper_segments(clip_path))
                progress.advance(task)

            pause_profile = self._average_dicts(pause_results)
            breath_profile = self._average_dicts(breath_results)
            filler_profile = self._average_dicts(filler_results)

            # Whisper threshold: mean confidence across all detected segments
            all_whisper_confs = [
                seg["confidence"]
                for segs in whisper_segments_all
                for seg in segs
            ]
            whisper_threshold = (
                float(np.mean(all_whisper_confs)) if all_whisper_confs else 0.0
            )

            # Extract voice quality separately for the profile
            voice_quality = prosody.extract_voice_quality(tmp_clips[0]) if tmp_clips else {}

            # ── Step 6: Emotion fingerprint ─────────────────────────────
            task = progress.add_task("Analysing emotions …", total=None)
            emotion_fp = emotion.build_emotion_fingerprint(processed)
            progress.update(task, completed=1, total=1)

            # ── Cleanup temp clips ──────────────────────────────────────
            for tmp in tmp_clips:
                tmp.unlink(missing_ok=True)

        # ── Step 7: Assemble profile ────────────────────────────────────
        profile = BehavioralProfile(
            speaker_id=speaker_id,
            speaker_embedding=embedding,
            reference_audio_path=str(ref_path),
            prosody=averaged_prosody,
            pause_profile=pause_profile,
            breathing_profile=breath_profile,
            filler_profile=filler_profile,
            emotion_fingerprint=emotion_fp,
            whisper_threshold=whisper_threshold,
            voice_quality=voice_quality,
            language=language,
        )

        logger.info("Profile built for '%s'", speaker_id)
        return profile

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_profile(
        profile: BehavioralProfile, path: Union[str, Path]
    ) -> Path:
        """Serialise a profile to JSON.

        Args:
            profile: The profile to save.
            path: Destination path (parent dirs created automatically).

        Returns:
            Resolved path to the saved JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved profile → %s", path.name)
        return path.resolve()

    @staticmethod
    def load_profile(path: Union[str, Path]) -> BehavioralProfile:
        """Load a profile from a JSON file.

        Args:
            path: Path to a JSON profile file.

        Returns:
            Reconstructed ``BehavioralProfile``.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded profile from %s", path.name)
        return BehavioralProfile.from_dict(data)

    # ------------------------------------------------------------------ #
    #  Pretty-print summary                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def summarize(profile: BehavioralProfile) -> None:
        """Print a rich, colour-coded summary table to the console.

        This is a quick way to inspect a profile without opening the JSON.
        """
        table = Table(
            title=f"🎙  Speaker Profile: {profile.speaker_id}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Category", style="cyan", width=22)
        table.add_column("Metric", style="white", width=28)
        table.add_column("Value", style="green", width=30)

        # ── Identity ────────────────────────────────────────────────────
        table.add_row("Identity", "Speaker ID", profile.speaker_id)
        table.add_row("Identity", "Language", profile.language)
        table.add_row("Identity", "Embedding shape", str(profile.speaker_embedding.shape))
        table.add_row("Identity", "Reference clip", str(Path(profile.reference_audio_path).name))
        table.add_row("Identity", "Created", profile.created_at)

        table.add_section()

        # ── Prosody ─────────────────────────────────────────────────────
        p = profile.prosody
        table.add_row("Prosody", "F0 mean (Hz)", f"{p.get('f0_mean', 0):.1f}")
        table.add_row("Prosody", "F0 range (Hz)", f"{p.get('f0_range', 0):.1f}")
        table.add_row("Prosody", "Intensity mean (dB)", f"{p.get('intensity_mean', 0):.1f}")
        table.add_row("Prosody", "Syllables/sec", f"{p.get('syllables_per_second', 0):.2f}")
        table.add_row("Prosody", "Words/min", f"{p.get('words_per_minute', 0):.0f}")

        table.add_section()

        # ── Voice quality ───────────────────────────────────────────────
        vq = profile.voice_quality
        table.add_row("Voice Quality", "Jitter (local)", f"{vq.get('jitter', 0):.6f}")
        table.add_row("Voice Quality", "Shimmer (local)", f"{vq.get('shimmer', 0):.6f}")
        table.add_row("Voice Quality", "HNR (dB)", f"{vq.get('hnr', 0):.1f}")

        table.add_section()

        # ── Pauses ──────────────────────────────────────────────────────
        pp = profile.pause_profile
        table.add_row("Pauses", "Mean pause (s)", f"{pp.get('mean_pause', 0):.3f}")
        table.add_row("Pauses", "Long pause threshold", f"{pp.get('long_pause_threshold', 0):.3f}")
        table.add_row("Pauses", "Clause-pause prob", f"{pp.get('clause_pause_prob', 0):.2%}")

        table.add_section()

        # ── Breathing ──────────────────────────────────────────────────
        bp = profile.breathing_profile
        table.add_row("Breathing", "Breaths/min", f"{bp.get('breath_frequency', 0):.1f}")
        table.add_row("Breathing", "Pre-sentence prob", f"{bp.get('pre_sentence_breath_prob', 0):.2%}")
        table.add_row("Breathing", "Audibility", f"{bp.get('breath_audibility', 0):.2f}")

        table.add_section()

        # ── Fillers ─────────────────────────────────────────────────────
        fp = profile.filler_profile
        table.add_row("Fillers", "Frequency (per 100 w)", f"{fp.get('filler_frequency', 0):.1f}")
        filler_list = fp.get("filler_words", [])
        table.add_row("Fillers", "Detected fillers", ", ".join(filler_list[:8]) or "—")

        table.add_section()

        # ── Emotion ─────────────────────────────────────────────────────
        ef = profile.emotion_fingerprint
        table.add_row("Emotion", "Dominant", ef.get("dominant_emotion", "—"))
        table.add_row("Emotion", "Emotional range", f"{ef.get('emotional_range', 0):.3f}")
        table.add_row("Emotion", "Baseline arousal", f"{ef.get('baseline_arousal', 0):.3f}")
        table.add_row("Emotion", "Baseline valence", f"{ef.get('baseline_valence', 0):.3f}")

        dist = ef.get("emotion_distribution", {})
        top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        for label, pct in top3:
            table.add_row("Emotion", f"  {label}", f"{pct:.1%}")

        table.add_section()

        # ── Whisper ─────────────────────────────────────────────────────
        table.add_row("Whisper", "Threshold", f"{profile.whisper_threshold:.3f}")

        console.print(table)

    # ------------------------------------------------------------------ #
    #  Utility                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _average_dicts(dicts: list[dict]) -> dict:
        """Average numeric values across a list of identically-keyed dicts.

        Non-numeric values (lists, strings, arrays) are taken from the
        first dict to keep the structure intact.
        """
        if not dicts:
            return {}
        if len(dicts) == 1:
            return dicts[0]

        merged: dict = {}
        for key in dicts[0]:
            values = [d[key] for d in dicts if key in d]
            sample = values[0]

            if isinstance(sample, (int, float)):
                merged[key] = float(np.mean(values))
            elif isinstance(sample, np.ndarray):
                # Average arrays of the same length; otherwise keep first
                try:
                    merged[key] = np.mean(values, axis=0)
                except Exception:
                    merged[key] = sample
            elif isinstance(sample, list) and values and all(
                isinstance(v, list) for v in values
            ):
                # For numeric lists (e.g. histograms), element-wise average
                if sample and isinstance(sample[0], (int, float)):
                    try:
                        merged[key] = np.mean(values, axis=0).tolist()
                    except Exception:
                        merged[key] = sample
                else:
                    # For lists of dicts/strings, concatenate
                    merged[key] = [item for sublist in values for item in sublist]
            else:
                merged[key] = sample

        return merged


# ────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Smoke-test the BehavioralProfile dataclass serialisation round-trip.
    
    A full pipeline test (build_from_directory) requires real models and
    audio, so we only exercise the data layer here.
    """
    console.rule("[bold green]ProfileBuilder — smoke test")

    # Build a dummy profile with synthetic data
    dummy = BehavioralProfile(
        speaker_id="test_speaker",
        speaker_embedding=np.random.randn(192).astype(np.float64),
        reference_audio_path=str(config.PROFILES_DIR / "test_speaker" / "reference.wav"),
        prosody={
            "f0_mean": 180.5, "f0_std": 30.2, "f0_min": 100.0,
            "f0_max": 350.0, "f0_range": 250.0,
            "intensity_mean": 65.0, "intensity_std": 8.0,
            "syllables_per_second": 4.5, "words_per_minute": 150.0,
            "articulation_rate": 5.2,
        },
        pause_profile={
            "pause_durations": [0.3, 0.5, 0.8],
            "mean_pause": 0.53,
            "pause_histogram": [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "pre_sentence_pause_mean": 0.65,
            "mid_sentence_pause_mean": 0.3,
            "clause_pause_prob": 0.33,
            "long_pause_threshold": 0.78,
        },
        breathing_profile={
            "breath_events": [],
            "breath_frequency": 12.0,
            "pre_sentence_breath_prob": 0.4,
            "breath_audibility": 0.25,
        },
        filler_profile={
            "filler_words": ["um", "like"],
            "filler_frequency": 3.5,
            "filler_positions": ["sentence_start", "mid_sentence"],
        },
        emotion_fingerprint={
            "dominant_emotion": "neutral",
            "emotion_distribution": {
                "neutral": 0.6, "happy": 0.2, "sad": 0.1,
                "angry": 0.05, "fearful": 0.02, "surprised": 0.02,
                "disgusted": 0.01,
            },
            "emotional_range": 0.65,
            "baseline_arousal": 0.35,
            "baseline_valence": 0.52,
        },
        whisper_threshold=0.15,
        voice_quality={"jitter": 0.012, "shimmer": 0.035, "hnr": 18.5},
        language="en",
    )

    # Test serialisation round-trip
    console.print("[cyan]Testing to_dict / from_dict round-trip …")
    d = dummy.to_dict()
    assert isinstance(d["speaker_embedding"], list), "Embedding should be a list"
    reconstructed = BehavioralProfile.from_dict(d)
    assert np.allclose(dummy.speaker_embedding, reconstructed.speaker_embedding)
    console.print("  ✓ Round-trip matches")

    # Test JSON persistence
    tmp_json = config.PROFILES_DIR / "_profile_test.json"
    builder = ProfileBuilder()
    builder.save_profile(dummy, tmp_json)
    loaded = builder.load_profile(tmp_json)
    assert loaded.speaker_id == dummy.speaker_id
    assert np.allclose(loaded.speaker_embedding, dummy.speaker_embedding)
    tmp_json.unlink(missing_ok=True)
    console.print("  ✓ JSON save/load matches")

    # Test summary display
    console.print("[cyan]Displaying profile summary …")
    builder.summarize(dummy)

    console.print("[bold green]✓ All ProfileBuilder tests passed")
