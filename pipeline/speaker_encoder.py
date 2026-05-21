"""
Speaker embedding extraction via SpeechBrain ECAPA-TDNN.

A speaker embedding is a compact numerical fingerprint of *who* is speaking,
invariant to *what* they say.  XTTS uses this embedding to condition its
decoder, so the quality of the clone depends heavily on getting a clean,
representative embedding.  We load the model once and keep it warm.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

import config

logger = logging.getLogger(__name__)

# SpeechBrain's ECAPA-TDNN expects exactly 16 kHz input
_ECAPA_SAMPLE_RATE: int = 16_000


class SpeakerEncoder:
    """Produce 192-dimensional speaker embeddings using ECAPA-TDNN.

    The model is loaded **once** in ``__init__`` and held on *config.DEVICE*
    for the lifetime of this object so repeated calls avoid cold-start overhead.
    """

    def __init__(self) -> None:
        """Load the ECAPA-TDNN model from HuggingFace / local cache.

        The ``savedir`` is set to ``config.MODELS_DIR / "spkrec-ecapa"`` so
        that subsequent runs don't re-download weights.
        """
        logger.info(
            "Loading speaker encoder (%s) on %s …",
            config.SPEAKER_ENCODER_MODEL,
            config.DEVICE,
        )
        save_dir = config.MODELS_DIR / "spkrec-ecapa"
        save_dir.mkdir(parents=True, exist_ok=True)

        self._model = EncoderClassifier.from_hparams(
            source=config.SPEAKER_ENCODER_MODEL,
            savedir=str(save_dir),
            run_opts={"device": config.DEVICE},
        )
        logger.info("Speaker encoder ready.")

    # ------------------------------------------------------------------ #
    #  Core encoding                                                      #
    # ------------------------------------------------------------------ #

    def encode(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Compute a single 192-dim speaker embedding from a waveform.

        Internally resamples to 16 kHz if *sr* differs, because ECAPA-TDNN
        was trained on VoxCeleb at that rate and accuracy degrades otherwise.

        Args:
            waveform: 1-D float32 audio samples.
            sr: Sample rate of *waveform*.

        Returns:
            Numpy array of shape ``(192,)`` — L2-normalised embedding.
        """
        waveform_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        # Resample to 16 kHz when the input rate doesn't match
        if sr != _ECAPA_SAMPLE_RATE:
            logger.debug("Resampling %d → %d Hz for ECAPA", sr, _ECAPA_SAMPLE_RATE)
            waveform_t = torchaudio.functional.resample(
                waveform_t, orig_freq=sr, new_freq=_ECAPA_SAMPLE_RATE
            )

        waveform_t = waveform_t.to(config.DEVICE)

        # SpeechBrain returns shape (1, 1, 192) — squeeze to (192,)
        with torch.no_grad():
            embedding = self._model.encode_batch(waveform_t)

        emb_np: np.ndarray = embedding.squeeze().cpu().numpy()

        # L2-normalise so cosine similarity is a simple dot product
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm

        logger.debug("Embedding shape: %s, norm: %.4f", emb_np.shape, np.linalg.norm(emb_np))
        return emb_np

    def encode_multiple(
        self, audio_list: list[tuple[np.ndarray, int]]
    ) -> np.ndarray:
        """Average embedding across several clips for a more robust identity.

        Using multiple clips reduces the effect of transient noise or atypical
        utterances in any single recording.

        Args:
            audio_list: List of ``(waveform, sr)`` pairs.

        Returns:
            L2-normalised average embedding of shape ``(192,)``.

        Raises:
            ValueError: If *audio_list* is empty.
        """
        if not audio_list:
            raise ValueError("audio_list is empty — cannot compute embedding")

        embeddings: list[np.ndarray] = []
        for idx, (waveform, sr) in enumerate(audio_list):
            logger.debug("Encoding clip %d / %d", idx + 1, len(audio_list))
            embeddings.append(self.encode(waveform, sr))

        avg = np.mean(embeddings, axis=0)

        # Re-normalise after averaging
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        logger.info(
            "Averaged %d embeddings → shape %s", len(embeddings), avg.shape
        )
        return avg

    # ------------------------------------------------------------------ #
    #  Similarity                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two speaker embeddings.

        Because embeddings are L2-normalised, this is equivalent to a dot
        product.  Values close to 1.0 mean "same speaker".

        Args:
            emb1: Shape ``(192,)``.
            emb2: Shape ``(192,)``.

        Returns:
            Cosine similarity in ``[-1, 1]``.
        """
        dot = float(np.dot(emb1, emb2))
        denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-10
        return dot / denom

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_embedding(embedding: np.ndarray, path: Union[str, Path]) -> Path:
        """Save an embedding to a ``.npy`` file.

        Args:
            embedding: Shape ``(192,)`` numpy array.
            path: Destination path (parent dirs created automatically).

        Returns:
            Resolved path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), embedding)
        logger.info("Saved embedding → %s", path.name)
        return path.resolve()

    @staticmethod
    def load_embedding(path: Union[str, Path]) -> np.ndarray:
        """Load a previously-saved embedding from disk.

        Args:
            path: Path to a ``.npy`` file.

        Returns:
            Numpy array of shape ``(192,)``.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        embedding = np.load(str(path))
        logger.debug("Loaded embedding from %s — shape %s", path.name, embedding.shape)
        return embedding


# ────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    console.rule("[bold green]SpeakerEncoder — smoke test")

    encoder = SpeakerEncoder()

    # Synthetic test signal (white noise — not real speech, but exercises the path)
    sr = 16_000
    duration = 3.0
    np.random.seed(42)
    fake_audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1

    console.print("[cyan]Encoding a 3 s synthetic clip…")
    emb = encoder.encode(fake_audio, sr)
    console.print(f"  Embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")

    console.print("[cyan]Encoding multiple clips and averaging…")
    clips = [(fake_audio, sr), (fake_audio * 0.8, sr)]
    avg_emb = encoder.encode_multiple(clips)
    console.print(f"  Averaged shape: {avg_emb.shape}")

    sim = encoder.similarity(emb, avg_emb)
    console.print(f"  Self-similarity: {sim:.4f}")

    # Round-trip persistence test
    tmp = config.MODELS_DIR / "_encoder_test.npy"
    encoder.save_embedding(emb, tmp)
    loaded = encoder.load_embedding(tmp)
    assert np.allclose(emb, loaded), "Round-trip mismatch!"
    tmp.unlink(missing_ok=True)

    console.print("[bold green]✓ All SpeakerEncoder tests passed")
