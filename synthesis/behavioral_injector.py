"""
Behavioral injection engine — replaces traditional SSML markup.

WHY this exists:
    SSML is rigid and XML-heavy; worse, most neural TTS backends
    (XTTS, Bark) don't support it at all.  This module achieves
    the same goal — controlling pauses, fillers, breathing, and
    emotion — by translating a BehavioralProfile into a flat list
    of SynthSegments that the TTS engine and audio stitcher can
    render natively.

    Every decision (inject a breath? insert a filler?) is driven
    by the speaker's measured behavioural statistics, so the output
    sounds like *that* speaker, not a generic voice.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from rich.console import Console

import config
from synthesis.text_processor import TextProcessor, SynthSegment

logger = logging.getLogger(__name__)
console = Console()


class BehavioralInjector:
    """
    Converts text + BehavioralProfile → a SynthPlan (list[SynthSegment]).

    The plan encodes exactly what the TTS engine should render:
    speech, events (breaths/fillers), and pauses, in order.
    """

    def __init__(self) -> None:
        """
        Initialise with a shared TextProcessor.

        The TextProcessor is constructed here rather than passed
        in because the injector owns the NLP step.
        """
        self._text_processor = TextProcessor()
        logger.info("BehavioralInjector initialised")

    # ── Public API ──────────────────────────────────────────────────

    def build_synth_plan(
        self,
        text: str,
        profile: Any,
        emotion_override: str | None = None,
    ) -> list[SynthSegment]:
        """
        Build a full synthesis plan from text and a BehavioralProfile.

        Algorithm:
          1. Parse text → list[Sentence]
          2. For each sentence:
             a. PRE-SENTENCE — conditional breath + leading pause
             b. SENTENCE BODY — clause-level TTS segments with
                random filler injection
             c. POST-SENTENCE — fade-out on trailing-off speakers
          3. Return flat segment list

        @param text              — Raw text to synthesise
        @param profile           — BehavioralProfile instance
        @param emotion_override  — Force a specific emotion on all segments
        @returns                 — Ordered list of SynthSegments
        """
        if not text or not text.strip():
            logger.warning("build_synth_plan received empty text")
            return []

        sentences = self._text_processor.parse(text)
        if not sentences:
            return []

        # Extract profile sub-dicts with safe defaults
        breathing = getattr(profile, "breathing_profile", {}) or {}
        pause_prof = getattr(profile, "pause_profile", {}) or {}
        emotion_fp = getattr(profile, "emotion_fingerprint", None)
        filler_prof = getattr(profile, "filler_profile", {}) or {}
        # Filler stats live inside filler_profile dict, not as top-level attrs
        filler_freq: float = filler_prof.get("filler_frequency", 0.0)
        filler_words: list[str] = filler_prof.get("filler_words", config.FILLER_WORDS)
        # Trailing-off is a speaking habit detected during profiling
        trailing_off: bool = pause_prof.get("trailing_off", False)

        plan: list[SynthSegment] = []

        for sent in sentences:
            # ── a) PRE-SENTENCE ─────────────────────────────────
            pre_breath_prob = breathing.get("pre_sentence_breath_prob", 0.0)
            if pre_breath_prob > 0.5:
                plan.append(
                    SynthSegment(type="event", content="breath_inhale")
                )

            mean_pause = pause_prof.get("mean_pause", 0.0)
            if mean_pause > 0.3:
                plan.append(
                    SynthSegment(
                        type="pause",
                        content="",
                        duration_ms=int(mean_pause * 1000),
                    )
                )

            # ── b) SENTENCE BODY — clause by clause ─────────────
            clauses = sent.clauses if sent.clauses else [sent.text]
            emotion = emotion_override or sent.estimated_emotion

            # If profile carries a dominant emotion, prefer that
            if emotion_override is None and emotion_fp is not None:
                if isinstance(emotion_fp, dict):
                    dominant = emotion_fp.get("dominant_emotion")
                else:
                    dominant = getattr(emotion_fp, "dominant_emotion", None)
                if dominant:
                    emotion = dominant

            for clause_idx, clause_text in enumerate(clauses):
                clause_text = clause_text.strip()
                if not clause_text:
                    continue

                # Random filler injection before a clause
                if (
                    filler_freq > 2.0
                    and random.random() < config.FILLER_INJECTION_PROB
                    and filler_words
                ):
                    chosen_filler = random.choice(filler_words)
                    plan.append(
                        SynthSegment(type="event", content=f"filler_{chosen_filler}")
                    )
                    logger.debug("Injected filler '%s' before clause", chosen_filler)

                # The actual TTS segment
                plan.append(
                    SynthSegment(
                        type="tts",
                        content=clause_text,
                        emotion=emotion,
                    )
                )

                # Inter-clause pause (skip after the last clause)
                if clause_idx < len(clauses) - 1:
                    pause_ms = self.sample_from_histogram(pause_prof)
                    plan.append(
                        SynthSegment(type="pause", content="", duration_ms=pause_ms)
                    )

            # ── c) POST-SENTENCE ────────────────────────────────
            if trailing_off and plan:
                # Walk backwards to find the last tts segment and mark it
                for seg in reversed(plan):
                    if seg.type == "tts":
                        seg.params["fade_out"] = True
                        break

        logger.info(
            "Built synth plan: %d segments from %d sentences",
            len(plan),
            len(sentences),
        )
        return plan

    def sample_from_histogram(self, pause_profile: dict[str, Any]) -> int:
        """
        Sample a pause duration (ms) from the profile's 10-bin histogram.

        WHY a histogram rather than a single mean?
        Real speakers have multi-modal pause distributions
        (short hesitations + long thinking pauses).  Sampling
        preserves that natural variation.

        Falls back to a sensible 200 ms if the profile lacks
        histogram data.

        @param pause_profile — Dict with keys "histogram_counts" and
                               "histogram_edges" (10 bins)
        @returns             — Pause duration in milliseconds
        """
        counts = pause_profile.get("histogram_counts")
        edges = pause_profile.get("histogram_edges")

        if not counts or not edges or len(counts) < 1:
            # Fallback: use mean_pause or a sensible default
            mean = pause_profile.get("mean_pause", 0.2)
            return max(50, int(mean * 1000))

        # Normalise counts to probabilities
        total = sum(counts)
        if total == 0:
            return 200

        probs = [c / total for c in counts]

        # Pick a bin via weighted random choice
        bin_idx = random.choices(range(len(counts)), weights=probs, k=1)[0]

        # Sample uniformly within the chosen bin
        low = edges[bin_idx] if bin_idx < len(edges) else 0.1
        high = edges[bin_idx + 1] if (bin_idx + 1) < len(edges) else low + 0.1

        duration_s = random.uniform(low, high)
        # Clamp to reasonable range (50 ms – 3 s)
        duration_ms = int(max(50, min(3000, duration_s * 1000)))
        return duration_ms


# ── Standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass, field as dc_field
    from types import SimpleNamespace

    console.rule("[bold green]BehavioralInjector — Standalone Test")

    # Build a mock BehavioralProfile
    mock_emotion = SimpleNamespace(
        dominant_emotion="neutral",
        baseline_arousal=0.5,
    )
    mock_profile = SimpleNamespace(
        breathing_profile={
            "pre_sentence_breath_prob": 0.7,
        },
        pause_profile={
            "mean_pause": 0.4,
            "histogram_counts": [2, 5, 10, 8, 3, 1, 0, 0, 0, 0],
            "histogram_edges": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        },
        emotion_fingerprint=mock_emotion,
        filler_frequency=3.5,
        filler_words=["uh", "um", "like"],
        trailing_off=True,
    )

    injector = BehavioralInjector()
    test_text = (
        "I really love this idea, and I think we should pursue it. "
        "But honestly, I'm not sure about the timeline."
    )

    console.print(f"\n[yellow]Input:[/] {test_text}\n")

    plan = injector.build_synth_plan(test_text, mock_profile)

    console.print("[bold magenta]Synth Plan:[/]")
    for i, seg in enumerate(plan):
        extra = ""
        if seg.duration_ms:
            extra += f"  dur={seg.duration_ms}ms"
        if seg.params:
            extra += f"  params={seg.params}"
        console.print(f"  {i:>3d}. [{seg.type:>5}] {seg.content!r}  emotion={seg.emotion}{extra}")

    console.rule("[bold green]Done")
