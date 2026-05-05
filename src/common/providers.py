"""
providers.py
------------
Shared event provider abstraction for commentary demos.

This module separates "where events come from" from "how commentary is generated."

Supported providers:
    HistoricalReplayProvider
        Loads saved cricket or soccer event files and yields normalized events.

    TimedReplayProvider
        Wraps another provider and yields events with a delay between events,
        giving the demo a pseudo-live feel.

    LiveProviderStub
        Placeholder for a future real API integration. It proves the project
        has an architectural slot for live polling without pretending that a
        production live feed is already implemented.

Example:
    python -m src.common.providers --sport cricket --provider historical --source raw/1527575.json --max-events 3

    python -m src.common.providers --sport cricket --provider timed --source raw/1527575.json --delay 0.5 --max-events 3

    python -m src.common.providers --sport cricket --provider live
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


class ProviderNotConfiguredError(RuntimeError):
    """
    Raised when a provider exists architecturally but has not been configured.
    """


class EventProvider(Protocol):
    """
    Minimal provider protocol.

    Any future live provider only needs to implement iter_events().
    """

    sport: str
    source_name: str

    def iter_events(self) -> Iterator[dict[str, Any]]:
        ...


def _normalize_sport(sport: str) -> str:
    normalized = str(sport or "").strip().lower()

    if normalized not in {"cricket", "soccer"}:
        raise ValueError("sport must be either 'cricket' or 'soccer'.")

    return normalized


def _copy_with_provider_metadata(
    event: dict[str, Any],
    *,
    sport: str,
    provider: str,
    source_name: str,
    sequence_number: int,
) -> dict[str, Any]:
    """
    Return a shallow copy of an event with provider metadata attached.
    """
    event_copy = dict(event)

    event_copy.setdefault("sport", sport)
    event_copy["_provider"] = provider
    event_copy["_provider_source"] = source_name
    event_copy["_provider_sequence"] = sequence_number

    return event_copy


@dataclass
class HistoricalReplayProvider:
    """
    Load saved event data and yield events one at a time.

    Cricket source:
        Cricsheet JSON file loaded by src.cricket.load_events.load_match_events

    Soccer source:
        StatsBomb event JSON file loaded by src.soccer.load_soccer_events.load_soccer_match_events
    """

    sport: str
    source_path: str | Path

    @property
    def source_name(self) -> str:
        return str(self.source_path)

    def _load_events(self) -> list[dict[str, Any]]:
        sport = _normalize_sport(self.sport)
        source_path = Path(self.source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Event source file not found: {source_path}")

        if sport == "cricket":
            from src.cricket.load_events import load_match_events

            return load_match_events(source_path)

        if sport == "soccer":
            from src.soccer.load_soccer_events import load_soccer_match_events

            return load_soccer_match_events(source_path)

        raise ValueError(f"Unsupported sport: {self.sport}")

    def iter_events(self) -> Iterator[dict[str, Any]]:
        sport = _normalize_sport(self.sport)
        events = self._load_events()

        for index, event in enumerate(events, start=1):
            yield _copy_with_provider_metadata(
                event,
                sport=sport,
                provider="historical",
                source_name=self.source_name,
                sequence_number=index,
            )


@dataclass
class TimedReplayProvider:
    """
    Wrap another provider and emit events with a fixed delay.

    This gives saved data a pseudo-live behavior without changing the generator.
    """

    provider: EventProvider
    delay_seconds: float = 1.0
    sleep_before_first: bool = False

    @property
    def sport(self) -> str:
        return self.provider.sport

    @property
    def source_name(self) -> str:
        return self.provider.source_name

    def iter_events(self) -> Iterator[dict[str, Any]]:
        delay = max(0.0, float(self.delay_seconds or 0.0))

        for index, event in enumerate(self.provider.iter_events(), start=1):
            if delay > 0 and (self.sleep_before_first or index > 1):
                time.sleep(delay)

            event_copy = dict(event)
            event_copy["_provider"] = "timed_replay"
            event_copy["_provider_delay_seconds"] = delay
            event_copy["_provider_emitted_at"] = time.time()

            yield event_copy


@dataclass
class LiveProviderStub:
    """
    Placeholder for future live API polling.

    This is intentionally not a fake live API. It exists so the architecture can
    honestly say that live providers plug into the same iter_events() interface.
    """

    sport: str
    endpoint_name: str = "not_configured"

    @property
    def source_name(self) -> str:
        return f"live:{self.endpoint_name}"

    def iter_events(self) -> Iterator[dict[str, Any]]:
        sport = _normalize_sport(self.sport)

        raise ProviderNotConfiguredError(
            f"Live provider for {sport} is not configured. "
            "Add a real API client later by implementing iter_events() with polling "
            "or websocket consumption, then yield the same normalized event dictionaries."
        )

        yield  # Keeps this function typed as an iterator.


def build_provider(
    *,
    sport: str,
    provider_name: str,
    source_path: str | Path = "",
    delay_seconds: float = 0.0,
) -> EventProvider:
    """
    Build a provider from CLI-style inputs.
    """
    sport = _normalize_sport(sport)
    provider_name = str(provider_name or "historical").strip().lower()

    if provider_name == "historical":
        if not source_path:
            raise ValueError("source_path is required for historical provider.")
        return HistoricalReplayProvider(sport=sport, source_path=source_path)

    if provider_name == "timed":
        if not source_path:
            raise ValueError("source_path is required for timed provider.")
        base_provider = HistoricalReplayProvider(sport=sport, source_path=source_path)
        return TimedReplayProvider(base_provider, delay_seconds=delay_seconds)

    if provider_name == "live":
        return LiveProviderStub(sport=sport)

    raise ValueError("provider_name must be one of: historical, timed, live.")


def iter_limited_events(
    provider: EventProvider,
    *,
    max_events: int | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Yield events from a provider with an optional max count.
    """
    for index, event in enumerate(provider.iter_events(), start=1):
        if max_events is not None and index > max_events:
            break

        yield event


def _event_preview(event: dict[str, Any]) -> str:
    sport = str(event.get("sport", "")).lower()

    if sport == "cricket":
        return (
            f"cricket | innings={event.get('innings')} "
            f"over={event.get('over')}.{event.get('ball_in_over')} "
            f"{event.get('batter', '')} vs {event.get('bowler', '')}"
        )

    if sport == "soccer":
        return (
            f"soccer | {event.get('minute', 0)}:{int(event.get('second', 0) or 0):02d} "
            f"{event.get('team', '')} | {event.get('player', '')} | "
            f"{event.get('event_type_raw', '')}"
        )

    return str(event)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test historical, timed, and live-stub event providers."
    )
    parser.add_argument("--sport", choices=["cricket", "soccer"], required=True)
    parser.add_argument(
        "--provider",
        choices=["historical", "timed", "live"],
        default="historical",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Path to a saved event file. Required for historical/timed providers.",
    )
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-events", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    max_events = None if args.max_events == 0 else args.max_events

    provider = build_provider(
        sport=args.sport,
        provider_name=args.provider,
        source_path=args.source,
        delay_seconds=args.delay,
    )

    print(f"Provider: {args.provider}")
    print(f"Sport: {args.sport}")
    print(f"Source: {provider.source_name}")
    print("=" * 72)

    try:
        for event in iter_limited_events(provider, max_events=max_events):
            print(_event_preview(event))
    except ProviderNotConfiguredError as exc:
        print(str(exc))


if __name__ == "__main__":
    main()