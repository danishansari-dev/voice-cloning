"""
Audio loading, preprocessing, and reference-clip selection.

This module exists because every downstream pipeline stage (speaker encoding,
prosody extraction, emotion analysis) needs consistently formatted audio.
Centralising I/O here prevents each module from re-implementing format
conversion, resampling, and silence trimming.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf

import config

logger = logging.getLogger(__name__)

# Formats librosa can decode via soundfile / ffmpeg
_SUPPORTED_EXTENSIONS: set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class AudioLoader:
    """Load, preprocess, chunk, and select reference clips from audio files.

    All methods are stateless — no model weights are held in memory — so a
    single ``AudioLoader`` instance can safely be shared across threads.
    """

    # ------------------------------------------------------------------ #
    #  Loading                                                            #
    # ------------------------------------------------------------------ #

    def load_file(self, path: Union[str, Path]) -> tuple[np.ndarray, int]:
        """Load an audio file into a mono waveform + sample-rate pair.

        Args:
            path: Filesystem path to a .wav, .mp3, or .flac file.

        Returns:
            ``(waveform, sample_rate)`` where *waveform* is a 1-D float32
            numpy array and *sample_rate* is the native rate of the file.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file extension is unsupported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{path.suffix}'. "
                f"Supported: {_SUPPORTED_EXTENSIONS}"
            )

        logger.info("Loading audio: %s", path.name)
        # sr=None preserves the native sample rate
        waveform, sr = librosa.load(str(path), sr=None, mono=True)
        logger.debug("Loaded %s — %.2f s @ %d Hz", path.name, len(waveform) / sr, sr)
        return waveform, sr

    def load_directory(
        self, dir_path: Union[str, Path]
    ) -> list[tuple[np.ndarray, int]]:
        """Load every supported audio file inside *dir_path* (non-recursive).

        Args:
            dir_path: Directory to scan.

        Returns:
            List of ``(waveform, sample_rate)`` tuples, one per file, sorted
            alphabetically by filename for determinism.

        Raises:
            FileNotFoundError: If directory does not exist.
            ValueError: If no supported audio files are found.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        audio_files = sorted(
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        if not audio_files:
            raise ValueError(f"No supported audio files in {dir_path}")

        logger.info("Found %d audio file(s) in %s", len(audio_files), dir_path)
        results: list[tuple[np.ndarray, int]] = []
        for f in audio_files:
            try:
                results.append(self.load_file(f))
            except Exception:
                # Skip corrupt / unreadable files but keep going
                logger.warning("Skipping unreadable file: %s", f.name, exc_info=True)
        return results

    # ------------------------------------------------------------------ #
    #  Preprocessing                                                      #
    # ------------------------------------------------------------------ #

    def preprocess(
        self, waveform: np.ndarray, sr: int
    ) -> tuple[np.ndarray, int]:
        """Normalise audio into the format every downstream module expects.

        Steps applied in order:
        1. Resample to 22 050 Hz (XTTS native rate from config).
        2. Peak-normalise amplitude to [-1, 1].
        3. Trim leading/trailing silence below –40 dB.
        4. Convert to mono (no-op if already mono).

        Args:
            waveform: Raw audio samples (1-D or 2-D).
            sr: Current sample rate.

        Returns:
            ``(processed_waveform, target_sr)``
        """
        target_sr: int = config.SAMPLE_RATE_XTTS

        # Mono conversion — average channels if stereo
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)

        # Resample only when necessary
        if sr != target_sr:
            logger.debug("Resampling %d → %d Hz", sr, target_sr)
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

        # Peak-normalise to prevent clipping downstream
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak

        # Trim silence — threshold from config
        waveform, _ = librosa.effects.trim(
            waveform, top_db=abs(config.SILENCE_THRESHOLD_DB)
        )

        return waveform, target_sr

    # ------------------------------------------------------------------ #
    #  Chunking                                                           #
    # ------------------------------------------------------------------ #

    def split_into_chunks(
        self,
        waveform: np.ndarray,
        sr: int,
        chunk_duration: int = 30,
    ) -> list[np.ndarray]:
        """Split a waveform into fixed-length chunks for batch processing.

        The last chunk may be shorter than *chunk_duration*; it is included
        as-is rather than padded so that silence-detection metrics stay valid.

        Args:
            waveform: 1-D float32 audio.
            sr: Sample rate.
            chunk_duration: Length of each chunk in seconds.

        Returns:
            List of 1-D numpy arrays.
        """
        chunk_samples = chunk_duration * sr
        chunks = [
            waveform[i : i + chunk_samples]
            for i in range(0, len(waveform), chunk_samples)
        ]
        logger.debug(
            "Split %.1f s audio into %d chunk(s) of ≤%d s",
            len(waveform) / sr,
            len(chunks),
            chunk_duration,
        )
        return chunks

    # ------------------------------------------------------------------ #
    #  Saving                                                             #
    # ------------------------------------------------------------------ #

    def save_audio(
        self,
        waveform: np.ndarray,
        sr: int,
        output_path: Union[str, Path],
    ) -> Path:
        """Write a waveform to disk as a WAV file.

        Args:
            waveform: 1-D float32 audio.
            sr: Sample rate.
            output_path: Destination path (parent dirs created automatically).

        Returns:
            Resolved ``Path`` to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform, sr)
        logger.info("Saved audio → %s (%.2f s)", output_path.name, len(waveform) / sr)
        return output_path.resolve()

    # ------------------------------------------------------------------ #
    #  Reference clip selection                                           #
    # ------------------------------------------------------------------ #

    def get_best_reference_clip(
        self,
        audio_list: list[tuple[np.ndarray, int]],
        target_duration: float = 6.0,
    ) -> tuple[np.ndarray, int]:
        """Pick the cleanest clip of approximately *target_duration* seconds.

        XTTS zero-shot cloning works best with ~6 s of high-quality speech.
        We score candidate windows on three proxies:
        • **SNR estimate** — ratio of RMS energy to the quietest 10 % of frames.
        • **Speech activity** — fraction of frames with RMS above the median.
        • **Spectral stability** — low zero-crossing rate indicates pitched
          speech rather than noise/breath.

        The window with the highest composite score wins.

        Args:
            audio_list: List of ``(waveform, sr)`` pairs.
            target_duration: Desired clip length in seconds (default 6.0).

        Returns:
            ``(best_clip, sr)`` — the best window resampled to XTTS rate.

        Raises:
            ValueError: If all clips are shorter than 1 second.
        """
        if not audio_list:
            raise ValueError("audio_list is empty — nothing to select from")

        target_sr = config.SAMPLE_RATE_XTTS
        best_score: float = -np.inf
        best_clip: np.ndarray | None = None

        for waveform, sr in audio_list:
            # Preprocess to a consistent format first
            waveform, sr = self.preprocess(waveform, sr)
            target_samples = int(target_duration * sr)

            # Handle audio shorter than the target duration
            if len(waveform) < sr:
                # Skip clips shorter than 1 s — too short to be useful
                logger.debug("Skipping clip < 1 s")
                continue

            if len(waveform) <= target_samples:
                # Use the whole clip if it's shorter than target
                candidates = [waveform]
            else:
                # Slide a window with 50 % overlap for finer selection
                hop = target_samples // 2
                candidates = [
                    waveform[i : i + target_samples]
                    for i in range(0, len(waveform) - target_samples + 1, hop)
                ]

            for clip in candidates:
                score = self._score_clip(clip, sr)
                if score > best_score:
                    best_score = score
                    best_clip = clip

        if best_clip is None:
            raise ValueError("No clip ≥ 1 s found in the provided audio")

        logger.info(
            "Selected reference clip: %.2f s, score %.4f",
            len(best_clip) / target_sr,
            best_score,
        )
        return best_clip, target_sr

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_clip(clip: np.ndarray, sr: int) -> float:
        """Composite quality score — higher is better.

        Combines three cheap proxies instead of running a full speech-quality
        model, which would be too slow for window-level scanning.
        """
        # Frame-level RMS energy (short frames for granularity)
        frame_length = int(0.025 * sr)  # 25 ms
        hop_length = int(0.010 * sr)    # 10 ms
        rms = librosa.feature.rms(
            y=clip, frame_length=frame_length, hop_length=hop_length
        )[0]

        if rms.max() == 0:
            return -np.inf

        # SNR estimate: ratio of mean energy to the quietest 10 % of frames
        sorted_rms = np.sort(rms)
        noise_floor = np.mean(sorted_rms[: max(1, len(sorted_rms) // 10)])
        snr = np.mean(rms) / (noise_floor + 1e-10)

        # Speech activity ratio — frames louder than the median
        speech_activity = np.mean(rms > np.median(rms))

        # Zero-crossing rate — lower means more harmonic / pitched speech
        zcr = librosa.feature.zero_crossing_rate(
            clip, frame_length=frame_length, hop_length=hop_length
        )[0]
        zcr_penalty = np.mean(zcr)  # we want this LOW

        # Weighted combination (weights chosen empirically)
        score = (0.5 * snr) + (0.3 * speech_activity) - (0.2 * zcr_penalty)
        return float(score)


# ────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    loader = AudioLoader()
    console.rule("[bold green]AudioLoader — smoke test")

    # --- Test with a synthetic chirp signal ---
    sr = 22050
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Chirp from 200 Hz → 2 000 Hz so there's harmonic content
    synth = 0.5 * np.sin(2 * np.pi * np.linspace(200, 2000, len(t)) * t).astype(
        np.float32
    )

    console.print("[cyan]Preprocessing synthetic 10 s chirp…")
    processed, new_sr = loader.preprocess(synth, sr)
    console.print(f"  After preprocess: {len(processed)/new_sr:.2f} s @ {new_sr} Hz")

    console.print("[cyan]Splitting into 3 s chunks…")
    chunks = loader.split_into_chunks(processed, new_sr, chunk_duration=3)
    console.print(f"  Got {len(chunks)} chunk(s)")

    console.print("[cyan]Selecting best reference clip…")
    audio_pairs = [(processed, new_sr)]
    best, best_sr = loader.get_best_reference_clip(audio_pairs, target_duration=6.0)
    console.print(f"  Best clip: {len(best)/best_sr:.2f} s @ {best_sr} Hz")

    # Save round-trip test
    tmp_path = config.OUTPUT_DIR / "_audioloader_test.wav"
    saved = loader.save_audio(best, best_sr, tmp_path)
    console.print(f"  Saved to {saved}")
    reloaded, rl_sr = loader.load_file(saved)
    console.print(f"  Reloaded: {len(reloaded)/rl_sr:.2f} s @ {rl_sr} Hz")
    # Cleanup
    tmp_path.unlink(missing_ok=True)

    console.print("[bold green]✓ All AudioLoader tests passed")
