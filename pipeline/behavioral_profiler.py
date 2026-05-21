"""
Behavioural speech-habit profiling.

This module captures the *micro-behaviours* that make a person's speech
recognisable beyond timbre: pause timing, breathing cadence, filler-word
habits, whispering tendencies, and overall speaking style.  These features
drive the "behavioural injection" layer that inserts natural disfluencies
into the synthesised output.

Primary VAD: pyannote.audio (transformer-based, most accurate).
Fallback VAD: webrtcvad (lightweight C library, works without GPU/token).
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf

import config

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────
_VAD_SR: int = 16_000          # webrtcvad requires exactly 16 kHz
_VAD_FRAME_MS: int = 30        # 10, 20, or 30 ms frames for webrtcvad
_VAD_AGGRESSIVENESS: int = 2   # 0–3; 2 is a balanced default


class BehavioralProfiler:
    """Profile pause timing, breathing, fillers, whisper, and speaking style.

    On construction we attempt to load **pyannote.audio**'s VAD pipeline
    (requires a HuggingFace token with access).  If that fails — e.g. no
    GPU, no token, or the model isn't cached — we fall back to **webrtcvad**
    transparently.
    """

    def __init__(self) -> None:
        self._pyannote_pipeline = None
        self._use_pyannote: bool = False
        self._init_vad()

    # ------------------------------------------------------------------ #
    #  VAD initialisation                                                 #
    # ------------------------------------------------------------------ #

    def _init_vad(self) -> None:
        """Try pyannote first; fall back to webrtcvad if unavailable."""
        try:
            from pyannote.audio import Pipeline as PyannotePipeline

            token = config.HUGGINGFACE_TOKEN
            if not token:
                raise RuntimeError("HUGGINGFACE_TOKEN not set — skipping pyannote")

            logger.info("Loading pyannote VAD pipeline …")
            self._pyannote_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/voice-activity-detection",
                use_auth_token=token,
            )
            self._use_pyannote = True
            logger.info("pyannote VAD ready.")
        except Exception as exc:
            logger.warning("pyannote unavailable (%s); falling back to webrtcvad", exc)
            self._use_pyannote = False

    # ------------------------------------------------------------------ #
    #  Voice-activity detection (unified interface)                       #
    # ------------------------------------------------------------------ #

    def _get_speech_segments(
        self, audio_path: Union[str, Path]
    ) -> list[tuple[float, float]]:
        """Return a list of ``(start_sec, end_sec)`` speech segments.

        Dispatches to pyannote or webrtcvad depending on what loaded.
        """
        if self._use_pyannote:
            return self._vad_pyannote(audio_path)
        return self._vad_webrtcvad(audio_path)

    def _vad_pyannote(
        self, audio_path: Union[str, Path]
    ) -> list[tuple[float, float]]:
        """VAD via pyannote.audio — most accurate but heavier."""
        output = self._pyannote_pipeline(str(audio_path))
        segments: list[tuple[float, float]] = []
        for turn, _, _ in output.itertracks(yield_label=True):
            segments.append((turn.start, turn.end))
        return segments

    def _vad_webrtcvad(
        self, audio_path: Union[str, Path]
    ) -> list[tuple[float, float]]:
        """VAD via webrtcvad — lightweight fallback.

        webrtcvad needs 16 kHz, 16-bit signed PCM frames of 10/20/30 ms.
        We load the file with librosa, convert to int16, then pack into
        raw PCM frames.
        """
        import webrtcvad

        vad = webrtcvad.Vad(_VAD_AGGRESSIVENESS)
        waveform, _ = librosa.load(str(audio_path), sr=_VAD_SR, mono=True)

        # Convert float32 [-1, 1] → int16
        pcm_int16 = (waveform * 32767).astype(np.int16)
        frame_samples = int(_VAD_SR * _VAD_FRAME_MS / 1000)
        frame_bytes = frame_samples * 2  # 16-bit = 2 bytes/sample

        segments: list[tuple[float, float]] = []
        speech_start: float | None = None

        for i in range(0, len(pcm_int16) - frame_samples + 1, frame_samples):
            frame = pcm_int16[i : i + frame_samples].tobytes()
            is_speech = vad.is_speech(frame, sample_rate=_VAD_SR)
            t = i / _VAD_SR

            if is_speech and speech_start is None:
                speech_start = t
            elif not is_speech and speech_start is not None:
                segments.append((speech_start, t))
                speech_start = None

        # Close any open segment at end of file
        if speech_start is not None:
            segments.append((speech_start, len(pcm_int16) / _VAD_SR))

        return segments

    # ------------------------------------------------------------------ #
    #  Pause detection                                                    #
    # ------------------------------------------------------------------ #

    def detect_pauses(self, audio_path: Union[str, Path]) -> dict:
        """Detect and characterise pauses between speech segments.

        Pause patterns are one of the strongest behavioural fingerprints:
        some speakers insert long pauses before sentences while others
        barely breathe between clauses.

        Args:
            audio_path: Path to an audio file.

        Returns:
            Dict with ``pause_durations``, ``mean_pause``,
            ``pause_histogram`` (10-bin), ``pre_sentence_pause_mean``,
            ``mid_sentence_pause_mean``, ``clause_pause_prob``,
            ``long_pause_threshold`` (95th percentile).
        """
        segments = self._get_speech_segments(audio_path)

        if len(segments) < 2:
            logger.warning("Fewer than 2 speech segments — no pauses detectable")
            return self._empty_pause_dict()

        pause_durations: list[float] = []
        # Track which pauses are likely pre-sentence vs mid-sentence
        pre_sentence_pauses: list[float] = []
        mid_sentence_pauses: list[float] = []

        for i in range(1, len(segments)):
            gap = segments[i][0] - segments[i - 1][1]
            if gap < config.MIN_PAUSE_DURATION_S:
                continue
            pause_durations.append(gap)

            # Heuristic: pauses > 0.5 s are likely sentence boundaries
            if gap > 0.5:
                pre_sentence_pauses.append(gap)
            else:
                mid_sentence_pauses.append(gap)

        if not pause_durations:
            return self._empty_pause_dict()

        arr = np.array(pause_durations)
        histogram, _ = np.histogram(arr, bins=10)

        return {
            "pause_durations": pause_durations,
            "mean_pause": float(np.mean(arr)),
            "pause_histogram": histogram.tolist(),
            "pre_sentence_pause_mean": (
                float(np.mean(pre_sentence_pauses)) if pre_sentence_pauses else 0.0
            ),
            "mid_sentence_pause_mean": (
                float(np.mean(mid_sentence_pauses)) if mid_sentence_pauses else 0.0
            ),
            "clause_pause_prob": len(mid_sentence_pauses) / len(pause_durations),
            "long_pause_threshold": float(np.percentile(arr, 95)),
        }

    @staticmethod
    def _empty_pause_dict() -> dict:
        """Return a zeroed-out pause dict for edge-case paths."""
        return {
            "pause_durations": [],
            "mean_pause": 0.0,
            "pause_histogram": [0] * 10,
            "pre_sentence_pause_mean": 0.0,
            "mid_sentence_pause_mean": 0.0,
            "clause_pause_prob": 0.0,
            "long_pause_threshold": 0.0,
        }

    # ------------------------------------------------------------------ #
    #  Breathing detection                                                #
    # ------------------------------------------------------------------ #

    def detect_breathing(self, audio_path: Union[str, Path]) -> dict:
        """Detect audible breath events in non-speech regions.

        Breathing patterns contribute to perceived naturalness: some
        speakers take loud inhales before every sentence, others are
        nearly silent.

        Detection heuristic:
        • Find non-speech regions (from VAD).
        • Within those, look for short energy bursts (50–500 ms) that are
          above a minimum threshold — these are candidate breath sounds.
        • Classify as inhale (rising energy) or exhale (falling energy).

        Args:
            audio_path: Path to an audio file.

        Returns:
            Dict with ``breath_events``, ``breath_frequency``,
            ``pre_sentence_breath_prob``, ``breath_audibility``.
        """
        audio_path = Path(audio_path)
        waveform, sr = librosa.load(str(audio_path), sr=_VAD_SR, mono=True)
        segments = self._get_speech_segments(audio_path)
        total_duration = len(waveform) / sr

        if total_duration < 1.0:
            logger.warning("Audio too short for breathing analysis")
            return self._empty_breath_dict()

        # Build list of non-speech intervals
        nonspeech: list[tuple[float, float]] = []
        prev_end = 0.0
        for start, end in segments:
            if start - prev_end > 0.05:
                nonspeech.append((prev_end, start))
            prev_end = end
        if total_duration - prev_end > 0.05:
            nonspeech.append((prev_end, total_duration))

        mean_energy = float(np.mean(waveform ** 2)) + 1e-10
        breath_events: list[dict] = []

        for ns_start, ns_end in nonspeech:
            ns_dur = ns_end - ns_start
            # Breaths are typically 0.05–0.5 s
            if ns_dur < 0.05 or ns_dur > 2.0:
                continue

            s_idx = int(ns_start * sr)
            e_idx = int(ns_end * sr)
            segment = waveform[s_idx:e_idx]

            seg_energy = float(np.mean(segment ** 2))

            # Must be above noise floor but below speech energy
            if seg_energy < mean_energy * 0.01:
                continue
            if seg_energy > mean_energy * config.BREATH_ENERGY_THRESHOLD * 3:
                continue  # likely speech leakage, not a breath

            # Classify inhale vs exhale by energy trajectory
            mid = len(segment) // 2
            first_half_e = np.mean(segment[:mid] ** 2)
            second_half_e = np.mean(segment[mid:] ** 2) if mid > 0 else first_half_e
            breath_type = "inhale" if first_half_e < second_half_e else "exhale"

            breath_events.append({
                "start": float(ns_start),
                "end": float(ns_end),
                "type": breath_type,
            })

        # Derive summary stats
        n_events = len(breath_events)
        breath_freq = (n_events / total_duration) * 60.0 if total_duration > 0 else 0.0

        # Pre-sentence breath probability: breaths that occur right before speech
        pre_sentence_breaths = 0
        for evt in breath_events:
            for s_start, _ in segments:
                if 0 < (s_start - evt["end"]) < 0.3:
                    pre_sentence_breaths += 1
                    break
        pre_sentence_prob = pre_sentence_breaths / max(1, n_events)

        # Audibility: average energy of breath events relative to speech
        if breath_events:
            avg_breath_energy = np.mean([
                np.mean(
                    waveform[int(e["start"] * sr) : int(e["end"] * sr)] ** 2
                )
                for e in breath_events
            ])
            breath_audibility = float(min(1.0, avg_breath_energy / mean_energy))
        else:
            breath_audibility = 0.0

        return {
            "breath_events": breath_events,
            "breath_frequency": float(breath_freq),
            "pre_sentence_breath_prob": float(pre_sentence_prob),
            "breath_audibility": float(breath_audibility),
        }

    @staticmethod
    def _empty_breath_dict() -> dict:
        """Zeroed-out breathing dict for edge cases."""
        return {
            "breath_events": [],
            "breath_frequency": 0.0,
            "pre_sentence_breath_prob": 0.0,
            "breath_audibility": 0.0,
        }

    # ------------------------------------------------------------------ #
    #  Filler-word detection                                              #
    # ------------------------------------------------------------------ #

    def detect_fillers(
        self,
        audio_path: Union[str, Path],
        transcript: str | None = None,
    ) -> dict:
        """Detect filler words / hesitations.

        When a transcript is available we search it directly for known
        fillers.  Without a transcript we return an empty profile —
        acoustic-only filler detection requires a fine-tuned model that
        isn't part of this pipeline yet.

        Args:
            audio_path: Path to an audio file (used for duration calc).
            transcript: Optional verbatim transcript.

        Returns:
            Dict with ``filler_words``, ``filler_frequency``,
            ``filler_positions``.
        """
        audio_path = Path(audio_path)
        waveform, sr = librosa.load(str(audio_path), sr=_VAD_SR, mono=True)
        total_duration = len(waveform) / sr

        if transcript is None:
            logger.info("No transcript — filler detection skipped (needs text)")
            return {
                "filler_words": [],
                "filler_frequency": 0.0,
                "filler_positions": [],
            }

        words = transcript.lower().split()
        word_count = len(words)
        filler_set = set(config.FILLER_WORDS)

        found_fillers: list[str] = []
        positions: list[str] = []

        for idx, word in enumerate(words):
            # Strip punctuation for matching
            clean = word.strip(".,!?;:\"'()-")
            if clean in filler_set:
                found_fillers.append(clean)

                # Heuristic position classification
                if idx == 0 or (idx > 0 and words[idx - 1].endswith((".", "!", "?"))):
                    positions.append("sentence_start")
                else:
                    positions.append("mid_sentence")

        # Also match two-word fillers like "you know", "i mean"
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        for bg in bigrams:
            if bg in filler_set:
                found_fillers.append(bg)
                positions.append("mid_sentence")

        freq = (len(found_fillers) / max(1, word_count)) * 100.0

        return {
            "filler_words": found_fillers,
            "filler_frequency": float(freq),
            "filler_positions": positions,
        }

    # ------------------------------------------------------------------ #
    #  Whisper segment detection                                          #
    # ------------------------------------------------------------------ #

    def detect_whisper_segments(
        self, audio_path: Union[str, Path]
    ) -> list[dict]:
        """Detect segments where the speaker drops to a whisper.

        Detection criteria:
        • Frame energy < 20 % of mean energy (whispers are quiet).
        • Low spectral flux (whispers have a stable, noisy spectrum).
        • Minimum duration of 0.2 s to avoid false positives.

        Args:
            audio_path: Path to an audio file.

        Returns:
            List of dicts with ``start``, ``end``, ``confidence``.
        """
        waveform, sr = librosa.load(str(audio_path), sr=_VAD_SR, mono=True)

        # Frame-level energy
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        rms = librosa.feature.rms(
            y=waveform, frame_length=frame_length, hop_length=hop_length
        )[0]
        mean_rms = np.mean(rms) + 1e-10

        # Spectral flux (frame-to-frame change in magnitude spectrum)
        S = np.abs(librosa.stft(waveform, n_fft=frame_length, hop_length=hop_length))
        flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))
        mean_flux = np.mean(flux) + 1e-10

        # Align lengths (flux is 1 shorter than rms due to diff)
        min_len = min(len(rms), len(flux))
        rms = rms[:min_len]
        flux = flux[:min_len]

        # Identify whisper frames
        energy_mask = rms < (config.BREATH_ENERGY_THRESHOLD * mean_rms)
        flux_mask = flux < (0.5 * mean_flux)
        whisper_mask = energy_mask & flux_mask

        # Group contiguous whisper frames into segments
        segments: list[dict] = []
        in_whisper = False
        seg_start = 0

        for i, is_whisper in enumerate(whisper_mask):
            t = (i * hop_length) / sr
            if is_whisper and not in_whisper:
                seg_start = t
                in_whisper = True
            elif not is_whisper and in_whisper:
                seg_end = t
                if seg_end - seg_start >= 0.2:
                    # Confidence based on how far below the energy threshold
                    seg_rms = np.mean(rms[max(0, i - 5) : i])
                    confidence = float(1.0 - min(1.0, seg_rms / mean_rms))
                    segments.append({
                        "start": float(seg_start),
                        "end": float(seg_end),
                        "confidence": confidence,
                    })
                in_whisper = False

        # Close any open segment
        if in_whisper:
            seg_end = len(waveform) / sr
            if seg_end - seg_start >= 0.2:
                segments.append({
                    "start": float(seg_start),
                    "end": float(seg_end),
                    "confidence": 0.5,
                })

        logger.info("Detected %d whisper segment(s)", len(segments))
        return segments

    # ------------------------------------------------------------------ #
    #  Speaking-habit profile                                             #
    # ------------------------------------------------------------------ #

    def build_habit_profile(
        self,
        audio_path: Union[str, Path],
        transcript: str | None = None,
    ) -> dict:
        """Characterise high-level speaking habits.

        Args:
            audio_path: Path to an audio file.
            transcript: Optional verbatim transcript.

        Returns:
            Dict with ``speaking_style``, ``average_sentence_length``,
            ``trailing_off``, ``uptalk_tendency``.
        """
        audio_path = Path(audio_path)
        waveform, sr = librosa.load(str(audio_path), sr=_VAD_SR, mono=True)

        # --- Speaking style from pause pattern --------------------------
        segments = self._get_speech_segments(audio_path)
        pause_durs = []
        for i in range(1, len(segments)):
            gap = segments[i][0] - segments[i - 1][1]
            if gap >= config.MIN_PAUSE_DURATION_S:
                pause_durs.append(gap)

        mean_pause = float(np.mean(pause_durs)) if pause_durs else 0.0
        # Formal speakers tend to have longer, more deliberate pauses
        if mean_pause > 0.6:
            style = "formal"
        elif mean_pause < 0.3:
            style = "casual"
        else:
            style = "mixed"

        # --- Average sentence length (from transcript or segment count) -
        if transcript:
            sentences = [s.strip() for s in transcript.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            avg_sentence_len = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0
        else:
            # Approximate: each long pause ≈ a sentence boundary
            n_sentences = max(1, len([p for p in pause_durs if p > 0.5]) + 1)
            total_speech_dur = sum(e - s for s, e in segments)
            avg_sentence_len = total_speech_dur / n_sentences  # in seconds

        # --- Trailing off: sentences that fade in energy ----------------
        trailing_off = False
        if len(segments) >= 2:
            # Check the last few speech segments for energy decay
            last_seg = segments[-1]
            s_idx = int(last_seg[0] * sr)
            e_idx = int(last_seg[1] * sr)
            seg_audio = waveform[s_idx:e_idx]
            if len(seg_audio) > sr // 2:
                mid = len(seg_audio) // 2
                first_e = np.mean(seg_audio[:mid] ** 2)
                second_e = np.mean(seg_audio[mid:] ** 2)
                # Energy drops by > 50 % in the second half
                trailing_off = bool(second_e < first_e * 0.5)

        # --- Uptalk: rising pitch at the end of utterances ─────────────
        uptalk_count = 0
        total_checked = 0
        for start, end in segments:
            seg_dur = end - start
            if seg_dur < 0.5:
                continue
            total_checked += 1
            # Look at the last 0.3 s of each segment
            tail_start = max(0, int((end - 0.3) * sr))
            tail_end = int(end * sr)
            tail = waveform[tail_start:tail_end]
            if len(tail) < 256:
                continue
            # Quick F0 estimate via autocorrelation
            try:
                import parselmouth
                tail_snd = parselmouth.Sound(tail, sampling_frequency=sr)
                pitch = tail_snd.to_pitch()
                f0s = pitch.selected_array["frequency"]
                voiced = f0s[f0s > 0]
                if len(voiced) >= 4:
                    # Rising if last quarter is higher than first quarter
                    q1 = np.mean(voiced[: len(voiced) // 4])
                    q4 = np.mean(voiced[-len(voiced) // 4 :])
                    if q4 > q1 * 1.1:
                        uptalk_count += 1
            except Exception:
                pass

        uptalk_tendency = uptalk_count / max(1, total_checked)

        return {
            "speaking_style": style,
            "average_sentence_length": float(avg_sentence_len),
            "trailing_off": trailing_off,
            "uptalk_tendency": float(uptalk_tendency),
        }


# ────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    from rich.console import Console

    console = Console()
    console.rule("[bold green]BehavioralProfiler — smoke test")

    # Generate a synthetic speech-like signal with pauses
    sr = 16_000
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    tone = 0.4 * np.sin(
        2 * np.pi * 200 * np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    ).astype(np.float32)
    # Two "utterances" separated by a pause
    synth = np.concatenate([tone, silence, tone, silence, tone])

    tmp_path = config.OUTPUT_DIR / "_behavioral_test.wav"
    sf.write(str(tmp_path), synth, sr)

    profiler = BehavioralProfiler()

    console.print("[cyan]Detecting pauses…")
    pauses = profiler.detect_pauses(tmp_path)
    console.print(f"  Mean pause: {pauses['mean_pause']:.3f} s, "
                  f"count: {len(pauses['pause_durations'])}")

    console.print("[cyan]Detecting breathing…")
    breath = profiler.detect_breathing(tmp_path)
    console.print(f"  Breath events: {len(breath['breath_events'])}, "
                  f"freq: {breath['breath_frequency']:.1f}/min")

    console.print("[cyan]Detecting fillers (no transcript)…")
    fillers = profiler.detect_fillers(tmp_path)
    console.print(f"  Fillers found: {len(fillers['filler_words'])}")

    console.print("[cyan]Detecting fillers (with transcript)…")
    sample_transcript = "um so I was like you know going to the store and uh basically I mean it was fine"
    fillers_t = profiler.detect_fillers(tmp_path, transcript=sample_transcript)
    console.print(f"  Fillers: {fillers_t['filler_words']}, "
                  f"freq: {fillers_t['filler_frequency']:.1f} per 100 words")

    console.print("[cyan]Detecting whisper segments…")
    whispers = profiler.detect_whisper_segments(tmp_path)
    console.print(f"  Whisper segments: {len(whispers)}")

    console.print("[cyan]Building habit profile…")
    habits = profiler.build_habit_profile(tmp_path)
    console.print(f"  Style: {habits['speaking_style']}, "
                  f"uptalk: {habits['uptalk_tendency']:.2f}")

    tmp_path.unlink(missing_ok=True)
    console.print("[bold green]✓ All BehavioralProfiler tests passed")
