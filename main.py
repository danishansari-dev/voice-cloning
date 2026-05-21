"""
CLI entry point for the AI Voice Cloning System.

WHY this exists:
    Provides a clean, user-friendly command-line interface for the
    entire voice cloning pipeline — from profiling a speaker's vocal
    behaviour to synthesising speech that replicates their unique
    speaking patterns.

Uses Typer for CLI structure and Rich for beautiful terminal output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

import config

logger = logging.getLogger(__name__)
console = Console()

# ── Typer app ────────────────────────────────────────────────────────
app = typer.Typer(
    name="voice-cloner",
    help="AI Voice Cloning System — replicate a speaker's full vocal behaviour.",
    add_completion=False,
    rich_markup_mode="rich",
)


# ── Helper: lazy imports ─────────────────────────────────────────────
# Heavy ML imports are deferred so --help responds instantly

def _get_profile_builder():
    """Lazily import ProfileBuilder to avoid slow startup on --help."""
    from pipeline.profile_builder import ProfileBuilder
    return ProfileBuilder()


def _get_tts_engine():
    """Lazily import TTSEngine to avoid loading XTTS on every CLI call."""
    from synthesis.tts_engine import TTSEngine
    return TTSEngine()


def _load_profile(name: str):
    """
    Load a saved BehavioralProfile by speaker name.

    @param name — Speaker name (must match a directory under profiles/)
    @returns    — BehavioralProfile instance
    """
    from pipeline.profile_builder import ProfileBuilder

    profile_path = config.PROFILES_DIR / name / "profile.json"
    if not profile_path.exists():
        console.print(
            f"[red]✗ Profile not found:[/] {profile_path}\n"
            f"  Run [cyan]python main.py profile --audio-dir <dir> --name {name}[/] first."
        )
        raise typer.Exit(code=1)

    builder = ProfileBuilder()
    profile = builder.load_profile(profile_path)
    console.print(f"[green]✓ Loaded profile:[/] {name}")
    return profile


# ── Command: profile ─────────────────────────────────────────────────

@app.command()
def profile(
    audio_dir: str = typer.Option(
        ...,
        "--audio-dir",
        "-d",
        help="Directory containing reference audio files (.wav/.mp3/.flac)",
    ),
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Speaker name — used to save and retrieve the profile",
    ),
    lang: str = typer.Option(
        "en",
        "--lang",
        "-l",
        help="Language code (XTTS supports 17 languages)",
    ),
) -> None:
    """
    Analyse reference audio and build a full BehavioralProfile.

    This runs all five extraction stages (speaker embedding, prosody,
    behavioural profiling, emotion analysis) and saves the merged
    profile to profiles/<name>/profile.json.
    """
    console.print(
        Panel(
            f"[bold cyan]Building voice profile for:[/] [yellow]{name}[/]\n"
            f"[bold cyan]Audio directory:[/] {audio_dir}\n"
            f"[bold cyan]Language:[/] {lang}",
            title="🎙️  Voice Profile Builder",
            border_style="cyan",
        )
    )

    audio_path = Path(audio_dir)
    if not audio_path.exists() or not audio_path.is_dir():
        console.print(f"[red]✗ Audio directory not found:[/] {audio_path}")
        raise typer.Exit(code=1)

    # Check for audio files before starting heavy model loads
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = [
        f for f in audio_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    if not audio_files:
        console.print(
            f"[red]✗ No audio files found in:[/] {audio_path}\n"
            f"  Supported formats: {', '.join(sorted(audio_extensions))}"
        )
        raise typer.Exit(code=1)

    console.print(f"[dim]Found {len(audio_files)} audio file(s)[/]\n")

    builder = _get_profile_builder()

    try:
        built_profile = builder.build_from_directory(
            audio_dir=audio_path,
            speaker_id=name,
            language=lang,
        )
    except Exception as exc:
        console.print(f"[red]✗ Profile building failed:[/] {exc}")
        logger.exception("Profile building failed")
        raise typer.Exit(code=1)

    # Save profile
    profile_dir = config.PROFILES_DIR / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.json"

    builder.save_profile(built_profile, profile_path)
    console.print(f"\n[green]✓ Profile saved to:[/] {profile_path}")

    # Print summary
    console.print()
    builder.summarize(built_profile)

    console.print(
        Panel(
            f"[green]Profile ready![/] Use [cyan]python main.py speak --name {name} --text \"...\"[/]",
            border_style="green",
        )
    )


# ── Command: speak ───────────────────────────────────────────────────

@app.command()
def speak(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Speaker name (must have a saved profile)",
    ),
    text: str = typer.Option(
        ...,
        "--text",
        "-t",
        help="Text to synthesise",
    ),
    emotion: str = typer.Option(
        "neutral",
        "--emotion",
        "-e",
        help="Emotion override: neutral, happy, sad, angry, whisper",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output WAV path (default: output/<name>_output.wav)",
    ),
    play: bool = typer.Option(
        False,
        "--play",
        "-p",
        help="Play audio after synthesis",
    ),
) -> None:
    """
    Synthesise speech using a saved BehavioralProfile.

    Generates audio that replicates the speaker's timbre, pauses,
    breathing patterns, fillers, and emotional baseline.
    """
    loaded_profile = _load_profile(name)

    # Determine output path
    if output:
        out_path = Path(output)
    else:
        out_path = config.OUTPUT_DIR / f"{name}_output.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold cyan]Speaker:[/] {name}\n"
            f"[bold cyan]Emotion:[/] {emotion}\n"
            f"[bold cyan]Text:[/] {text[:80]}{'…' if len(text) > 80 else ''}\n"
            f"[bold cyan]Output:[/] {out_path}",
            title="🔊  Speech Synthesis",
            border_style="cyan",
        )
    )

    # Set emotion override only if not "neutral" (let profile's baseline flow through)
    emotion_override = emotion if emotion != "neutral" else None

    engine = _get_tts_engine()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesising speech…", total=None)

        try:
            audio = engine.synthesize_with_profile(
                text=text,
                profile=loaded_profile,
                emotion_override=emotion_override,
                output_path=out_path,
            )
        except Exception as exc:
            console.print(f"[red]✗ Synthesis failed:[/] {exc}")
            logger.exception("Synthesis failed")
            raise typer.Exit(code=1)

        progress.update(task, completed=True)

    duration_s = len(audio) / config.SAMPLE_RATE_UNIFIED if len(audio) > 0 else 0
    console.print(
        f"\n[green]✓ Audio generated:[/] {out_path}\n"
        f"[dim]  Duration: {duration_s:.2f}s | "
        f"Samples: {len(audio):,} | "
        f"Rate: {config.SAMPLE_RATE_UNIFIED} Hz[/]"
    )

    # Play audio if requested
    if play:
        _play_audio(out_path)


# ── Command: speak-file ──────────────────────────────────────────────

@app.command("speak-file")
def speak_file(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Speaker name (must have a saved profile)",
    ),
    input_file: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to text file to synthesise",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output WAV path (default: output/<name>_full.wav)",
    ),
    emotion: str = typer.Option(
        "neutral",
        "--emotion",
        "-e",
        help="Emotion override for all paragraphs",
    ),
    play: bool = typer.Option(
        False,
        "--play",
        "-p",
        help="Play audio after synthesis",
    ),
) -> None:
    """
    Synthesise speech from a text file, paragraph by paragraph.

    Each paragraph is synthesised separately and then concatenated
    with natural inter-paragraph pauses derived from the speaker's
    behavioural profile.
    """
    loaded_profile = _load_profile(name)

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]✗ Input file not found:[/] {input_path}")
        raise typer.Exit(code=1)

    # Read and split into paragraphs (double newline separated)
    raw_text = input_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

    if not paragraphs:
        console.print("[red]✗ Input file is empty or has no text[/]")
        raise typer.Exit(code=1)

    # Determine output path
    if output:
        out_path = Path(output)
    else:
        out_path = config.OUTPUT_DIR / f"{name}_full.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold cyan]Speaker:[/] {name}\n"
            f"[bold cyan]Input:[/] {input_path}\n"
            f"[bold cyan]Paragraphs:[/] {len(paragraphs)}\n"
            f"[bold cyan]Output:[/] {out_path}",
            title="📄  File Synthesis",
            border_style="cyan",
        )
    )

    emotion_override = emotion if emotion != "neutral" else None
    engine = _get_tts_engine()

    # Synthesise each paragraph and collect waveforms
    all_clips: list[np.ndarray] = []
    # Inter-paragraph pause — derived from profile's pause stats
    pause_profile = getattr(loaded_profile, "pause_profile", {}) or {}
    inter_para_pause_s = pause_profile.get("long_pause_threshold", 0.8)
    inter_para_silence = np.zeros(
        int(config.SAMPLE_RATE_UNIFIED * inter_para_pause_s),
        dtype=np.float32,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, para in enumerate(paragraphs):
            task = progress.add_task(
                f"Paragraph {i + 1}/{len(paragraphs)}: {para[:50]}…",
                total=None,
            )

            try:
                audio = engine.synthesize_with_profile(
                    text=para,
                    profile=loaded_profile,
                    emotion_override=emotion_override,
                )
                all_clips.append(audio)

                # Add inter-paragraph pause (except after the last paragraph)
                if i < len(paragraphs) - 1:
                    all_clips.append(inter_para_silence)

            except Exception as exc:
                console.print(
                    f"[yellow]⚠ Paragraph {i + 1} failed: {exc}[/]"
                )
                logger.warning("Paragraph %d failed: %s", i + 1, exc)

            progress.update(task, completed=True)

    if not all_clips:
        console.print("[red]✗ No audio was generated[/]")
        raise typer.Exit(code=1)

    # Concatenate all paragraphs
    from synthesis.audio_stitcher import AudioStitcher

    stitcher = AudioStitcher()
    final = np.concatenate(all_clips)
    final = stitcher.normalize(final, target_db=config.NORMALIZE_TARGET_DB)
    stitcher.save(final, config.SAMPLE_RATE_UNIFIED, out_path)

    duration_s = len(final) / config.SAMPLE_RATE_UNIFIED
    console.print(
        f"\n[green]✓ Full audio saved:[/] {out_path}\n"
        f"[dim]  Duration: {duration_s:.2f}s | "
        f"Paragraphs: {len(paragraphs)} | "
        f"Samples: {len(final):,}[/]"
    )

    if play:
        _play_audio(out_path)


# ── Command: list-profiles ───────────────────────────────────────────

@app.command("list-profiles")
def list_profiles() -> None:
    """
    List all saved speaker profiles.

    Scans the profiles/ directory for valid profile.json files
    and displays a summary table.
    """
    profiles_root = config.PROFILES_DIR

    if not profiles_root.exists():
        console.print("[yellow]No profiles directory found.[/]")
        raise typer.Exit(code=0)

    # Find all profile.json files
    profile_dirs = sorted([
        d for d in profiles_root.iterdir()
        if d.is_dir() and (d / "profile.json").exists()
    ])

    if not profile_dirs:
        console.print(
            Panel(
                "[yellow]No saved profiles found.[/]\n"
                "Run [cyan]python main.py profile --audio-dir <dir> --name <name>[/] to create one.",
                title="📋  Profiles",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=0)

    # Build rich table
    table = Table(
        title="🎙️  Saved Voice Profiles",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Speaker", style="bold white")
    table.add_column("Language", justify="center")
    table.add_column("Dominant Emotion", justify="center")
    table.add_column("Speaking Style", justify="center")
    table.add_column("Created", justify="center", style="dim")
    table.add_column("Reference Audio", style="dim")

    import json

    for profile_dir in profile_dirs:
        profile_path = profile_dir / "profile.json"
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))

            speaker_id = data.get("speaker_id", profile_dir.name)
            language = data.get("language", "en")
            created_at = data.get("created_at", "unknown")

            # Safely extract nested fields
            emotion_fp = data.get("emotion_fingerprint", {})
            dominant_emotion = emotion_fp.get("dominant_emotion", "—")

            # Habit profile may be nested in various ways
            pause_prof = data.get("pause_profile", {})
            speaking_style = pause_prof.get("speaking_style", "—")

            ref_audio = data.get("reference_audio_path", "—")
            if ref_audio and ref_audio != "—":
                ref_audio = Path(ref_audio).name  # Just the filename

            table.add_row(
                speaker_id,
                language,
                dominant_emotion,
                speaking_style,
                created_at[:10] if len(created_at) > 10 else created_at,
                ref_audio,
            )
        except Exception as exc:
            table.add_row(
                profile_dir.name,
                "?", "?", "?",
                "[red]error[/]",
                str(exc)[:30],
            )

    console.print(table)
    console.print(f"\n[dim]Profiles directory: {profiles_root}[/]")


# ── Helper: play audio ───────────────────────────────────────────────

def _play_audio(path: Path) -> None:
    """
    Play a WAV file using pydub + system audio.

    Falls back gracefully if pydub or audio playback isn't available
    (e.g., headless server).

    @param path — Path to the WAV file to play
    """
    try:
        from pydub import AudioSegment
        from pydub.playback import play

        console.print("[dim]Playing audio…[/]")
        audio = AudioSegment.from_wav(str(path))
        play(audio)
        console.print("[green]✓ Playback complete[/]")
    except ImportError:
        console.print(
            "[yellow]⚠ pydub not available for playback. "
            "Install with: pip install pydub[/]"
        )
    except Exception as exc:
        console.print(f"[yellow]⚠ Playback failed: {exc}[/]")
        logger.warning("Audio playback failed: %s", exc)


# ── Entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(
        Panel(
            "[bold white]AI Voice Cloning System[/]\n"
            "[dim]Fully open-source • XTTS-v2 + Bark • Zero-shot cloning[/]",
            border_style="bright_cyan",
            padding=(1, 4),
        )
    )
    app()
