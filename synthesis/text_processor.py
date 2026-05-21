"""
NLP-driven text processor for the synthesis pipeline.

WHY this exists:
    Raw input text must be decomposed into linguistically meaningful
    chunks (sentences → clauses) before we can inject behavioural cues
    like fillers, pauses, and emotion conditioning.  SpaCy gives us
    dependency-parsed sentences and clause boundaries so the downstream
    BehavioralInjector can make informed decisions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import spacy
from spacy.tokens import Span
from rich.console import Console

import config

logger = logging.getLogger(__name__)
console = Console()


# ── Emotion lexicon — intentionally tiny; real systems would use a model ─
_POSITIVE_WORDS: set[str] = {
    "happy", "glad", "wonderful", "great", "love", "excellent",
    "fantastic", "joy", "cheerful", "excited", "amazing", "beautiful",
    "awesome", "brilliant", "delighted", "thrilled", "pleased",
}

_NEGATIVE_WORDS: set[str] = {
    "sad", "angry", "hate", "terrible", "horrible", "awful",
    "disgusting", "furious", "upset", "miserable", "depressed",
    "annoyed", "frustrated", "painful", "worst", "dreadful",
}


# ── Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class Sentence:
    """
    Structured representation of a single parsed sentence.

    Carries enough linguistic detail for the injector to decide
    where to place pauses, fillers, and emotion shifts.
    """

    text: str
    tokens: list[str]
    clauses: list[str]
    dep_tree: list[dict[str, Any]]
    is_question: bool
    ends_with_comma: bool
    estimated_emotion: str
    word_count: int


@dataclass
class SynthSegment:
    """
    Atomic unit the TTS engine will render.

    Three segment types exist:
      • "tts"   — text to be voiced
      • "event" — paralinguistic event (breath, filler, laugh)
      • "pause" — pure silence of a given duration
    """

    type: str                     # "tts" | "event" | "pause"
    content: str                  # text for tts, event name for event
    duration_ms: int = 0          # for pause segments, 0 otherwise
    emotion: str = "neutral"      # for tts segments
    is_whisper: bool = False      # for tts segments
    params: dict[str, Any] = field(default_factory=dict)


# ── TextProcessor ───────────────────────────────────────────────────────

class TextProcessor:
    """
    Parses raw text into linguistically-annotated Sentences and
    converts them into SynthSegment lists ready for synthesis.

    SpaCy is loaded once at construction to avoid repeated model
    loading overhead.
    """

    def __init__(self) -> None:
        """Load spacy model once — avoids per-call loading overhead."""
        try:
            console.print("[cyan]Loading spaCy en_core_web_sm…[/]")
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully")
        except OSError:
            # Fallback: attempt download then reload
            logger.warning("spaCy model not found — attempting download")
            from spacy.cli import download  # noqa: WPS433
            download("en_core_web_sm")
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model downloaded and loaded")

    # ── Public API ──────────────────────────────────────────────────

    def parse(self, text: str) -> list[Sentence]:
        """
        Parse raw text into a list of Sentence dataclasses.

        @param text  — The raw input string to process
        @returns     — One Sentence per sentence detected by spaCy
        """
        if not text or not text.strip():
            logger.warning("parse() received empty text — returning empty list")
            return []

        doc = self._nlp(text)
        sentences: list[Sentence] = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            tokens = [token.text for token in sent]
            clause_boundary_indices = self.detect_clause_boundaries(sent)
            clauses = self._split_at_boundaries(sent, clause_boundary_indices)

            dep_tree = [
                {
                    "text": token.text,
                    "dep": token.dep_,
                    "head": token.head.text,
                    "pos": token.pos_,
                }
                for token in sent
            ]

            sentences.append(
                Sentence(
                    text=sent_text,
                    tokens=tokens,
                    clauses=clauses,
                    dep_tree=dep_tree,
                    is_question=sent_text.endswith("?"),
                    ends_with_comma=sent_text.endswith(","),
                    estimated_emotion=self.estimate_sentence_emotion(sent_text),
                    word_count=len([t for t in sent if not t.is_punct]),
                )
            )

        logger.info("Parsed %d sentence(s) from %d-char input", len(sentences), len(text))
        return sentences

    def detect_clause_boundaries(self, sentence: Span) -> list[int]:
        """
        Find token indices where clause boundaries occur.

        WHY: Clause breaks are natural pause insertion points.
        We look for coordinating conjunctions (cc), subordinating
        conjunctions (mark), commas, and relative clauses (relcl).

        @param sentence — A spaCy Span representing one sentence
        @returns        — Sorted list of token indices (within the span)
        """
        boundaries: list[int] = []
        # Offset so indices are relative to the span, not the doc
        span_start = sentence.start

        for token in sentence:
            relative_idx = token.i - span_start

            # Coordinating conjunction (e.g. "and", "but", "or")
            if token.dep_ == "cc":
                boundaries.append(relative_idx)

            # Subordinating conjunction / complementiser (e.g. "because", "although")
            elif token.dep_ == "mark":
                boundaries.append(relative_idx)

            # Comma — common clause separator in English
            elif token.text == ",":
                boundaries.append(relative_idx)

            # Relative clause attachment point
            elif token.dep_ == "relcl":
                boundaries.append(relative_idx)

        # Deduplicate and sort
        return sorted(set(boundaries))

    def estimate_sentence_emotion(self, sentence: str) -> str:
        """
        Quick lexicon-based emotion estimate.

        WHY: A fast heuristic avoids running a full emotion model
        during text pre-processing; the real emotion conditioning
        happens downstream in EmotionConditioner.

        @param sentence — Raw sentence text
        @returns        — One of "happy", "angry", "sad", "neutral"
        """
        lower_tokens = set(re.findall(r"\b\w+\b", sentence.lower()))

        pos_count = len(lower_tokens & _POSITIVE_WORDS)
        neg_count = len(lower_tokens & _NEGATIVE_WORDS)

        if pos_count > neg_count:
            return "happy"
        elif neg_count > pos_count:
            # Simple split: many negative tokens → angry, few → sad
            return "angry" if neg_count >= 2 else "sad"
        return "neutral"

    def segment_for_synthesis(self, text: str) -> list[SynthSegment]:
        """
        Convert raw text into an ordered list of SynthSegments.

        WHY: The TTS engine and audio stitcher need a flat, typed
        list of segments (speech / event / pause) rather than raw text.

        @param text — The full text to synthesise
        @returns    — Flat list of SynthSegments with emotion hints
        """
        if not text or not text.strip():
            return []

        sentences = self.parse(text)
        segments: list[SynthSegment] = []

        for idx, sent in enumerate(sentences):
            # Each clause becomes a TTS segment with the sentence emotion
            for clause in sent.clauses:
                clause_text = clause.strip()
                if not clause_text:
                    continue

                segments.append(
                    SynthSegment(
                        type="tts",
                        content=clause_text,
                        emotion=sent.estimated_emotion,
                    )
                )

            # Inter-sentence pause (skip after the last sentence)
            if idx < len(sentences) - 1:
                segments.append(
                    SynthSegment(type="pause", content="", duration_ms=250)
                )

        logger.info("Generated %d synth segments from text", len(segments))
        return segments

    # ── Private helpers ─────────────────────────────────────────────

    def _split_at_boundaries(self, sentence: Span, boundaries: list[int]) -> list[str]:
        """
        Split sentence text into clause chunks at the given token indices.

        Boundary tokens (commas, conjunctions) are kept with the
        preceding clause to preserve natural reading flow.
        """
        if not boundaries:
            return [sentence.text.strip()]

        tokens = list(sentence)
        clauses: list[str] = []
        prev = 0

        for boundary_idx in boundaries:
            if boundary_idx <= prev:
                continue

            chunk_tokens = tokens[prev:boundary_idx]
            chunk_text = " ".join(t.text for t in chunk_tokens).strip()
            if chunk_text:
                clauses.append(chunk_text)
            prev = boundary_idx

        # Remaining tokens after the last boundary
        remaining = tokens[prev:]
        remaining_text = " ".join(t.text for t in remaining).strip()
        if remaining_text:
            clauses.append(remaining_text)

        # Edge case: if splitting produced nothing, return the whole sentence
        return clauses if clauses else [sentence.text.strip()]


# ── Standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    console.rule("[bold green]TextProcessor — Standalone Test")

    tp = TextProcessor()

    test_text = (
        "I love this wonderful day, and the sun is shining brightly. "
        "Why does everything feel so terrible? "
        "He said he was happy, but I think he was lying."
    )

    console.print(f"\n[yellow]Input:[/] {test_text}\n")

    # Test parsing
    sentences = tp.parse(test_text)
    for i, s in enumerate(sentences):
        console.print(f"[cyan]Sentence {i + 1}:[/] {s.text}")
        console.print(f"  Tokens:   {s.tokens}")
        console.print(f"  Clauses:  {s.clauses}")
        console.print(f"  Question: {s.is_question}")
        console.print(f"  Emotion:  {s.estimated_emotion}")
        console.print(f"  Words:    {s.word_count}")
        console.print()

    # Test segmentation
    segments = tp.segment_for_synthesis(test_text)
    console.print("[bold magenta]Synthesis Segments:[/]")
    for seg in segments:
        console.print(f"  [{seg.type:>5}] {seg.content!r}  emotion={seg.emotion}  dur={seg.duration_ms}ms")

    console.rule("[bold green]Done")
