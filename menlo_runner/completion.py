from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


LEVEL_DELIVERY_POINTS = {
    0: 10,
    1: 20,
    2: 30,
}


@dataclass(frozen=True)
class CompletionConfig:
    """Stop conditions and scoring settings for a project completion run."""

    level: int | None = None
    points_per_delivery: int | None = None
    max_delivered_cubes: int | None = None
    max_elapsed_s: float | None = None

    def validate(self) -> None:
        if self.level is not None and self.level not in LEVEL_DELIVERY_POINTS:
            raise ValueError("level must be 0, 1, 2, or None.")
        if self.points_per_delivery is not None and self.points_per_delivery < 0:
            raise ValueError("points_per_delivery must be zero or greater.")
        if self.max_delivered_cubes is not None and self.max_delivered_cubes < 0:
            raise ValueError("max_delivered_cubes must be zero or greater.")
        if self.max_elapsed_s is not None and self.max_elapsed_s <= 0:
            raise ValueError("max_elapsed_s must be greater than zero.")
        if self.max_delivered_cubes is None and self.max_elapsed_s is None:
            raise ValueError("Set max_delivered_cubes, max_elapsed_s, or both.")

    def delivery_points(self) -> int:
        if self.points_per_delivery is not None:
            return self.points_per_delivery
        if self.level is None:
            return 0
        return LEVEL_DELIVERY_POINTS[self.level]


class CompletionTracker:
    """Measure a run from the first agent cycle and report completion reasons."""

    def __init__(
        self,
        config: CompletionConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        config.validate()
        self.config = config
        self._clock = clock
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.end_reason: str | None = None
        self._active_cube_ids: set[str] = set()
        self._delivered_cube_ids: set[str] = set()

    def start_first_cycle(self) -> None:
        if self.started_at is None:
            self.started_at = self._clock()

    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        stop_at = self.ended_at if self.ended_at is not None else self._clock()
        return max(0.0, stop_at - self.started_at)

    def stop_reason(self, delivered_count: int) -> str | None:
        if (
            self.config.max_delivered_cubes is not None
            and delivered_count >= self.config.max_delivered_cubes
        ):
            return f"delivered {delivered_count}/{self.config.max_delivered_cubes} cubes"
        if self.config.max_elapsed_s is not None and self.elapsed_s() >= self.config.max_elapsed_s:
            return f"elapsed {self.elapsed_s():.1f}/{self.config.max_elapsed_s:.1f} seconds"
        return None

    async def scene_delivered_count(self, ctx: Any) -> int:
        """Track delivered cubes from scene-state transitions.

        The simulator keeps hidden reserve cubes in ``scene_state`` as
        ``visible=False``. Counting all invisible cubes inflates delivery
        progress, so this tracker only counts a cube after it has first been
        active in the run (visible or held) and later becomes invisible.
        """
        scene = await ctx.state("scene_state")

        for entity_key, entity in scene.entities.items():
            entity_id = getattr(entity, "entity_id", entity_key)
            if not isinstance(entity_id, str) or not entity_id.startswith("cube_"):
                continue

            visible = bool(getattr(entity, "visible", False))
            attached_to = getattr(entity, "attached_to", None)
            is_active = visible or attached_to is not None

            if is_active:
                self._active_cube_ids.add(entity_id)
                continue

            if entity_id in self._active_cube_ids:
                self._delivered_cube_ids.add(entity_id)

        return len(self._delivered_cube_ids)

    async def stop_reason_from_scene(self, ctx: Any) -> str | None:
        """Check stop conditions using authoritative scene progress."""
        return self.stop_reason(await self.scene_delivered_count(ctx))

    def delivery_score(self, delivered_count: int) -> int:
        return delivered_count * self.config.delivery_points()

    def mark_ended(self, reason: str) -> None:
        if self.ended_at is None:
            self.ended_at = self._clock()
            self.end_reason = reason

    def print_start(self) -> None:
        target_cubes = (
            self.config.max_delivered_cubes
            if self.config.max_delivered_cubes is not None
            else "any"
        )
        time_limit = self.config.max_elapsed_s if self.config.max_elapsed_s is not None else "none"
        delivery_points = self.config.delivery_points()
        score_text = (
            f"{delivery_points} points per delivery"
            if delivery_points
            else "delivery scoring not configured"
        )
        print(
            "Completion timer started at first cycle "
            f"(target cubes={target_cubes}, "
            f"time limit={time_limit}s, "
            f"{score_text})."
        )

    def print_summary(self, delivered_count: int) -> None:
        reason = self.end_reason or self.stop_reason(delivered_count) or "agent stopped"
        print(
            "Completion run ended: "
            f"{reason}; elapsed={self.elapsed_s():.1f}s; delivered={delivered_count}; "
            f"delivery_score={self.delivery_score(delivered_count)}."
        )

    async def print_summary_from_scene(self, ctx: Any) -> None:
        """Print the completion summary using authoritative scene progress."""
        self.print_summary(await self.scene_delivered_count(ctx))


def level_from_program_name(program_name: str) -> int | None:
    if "level-0" in program_name or "level_0" in program_name:
        return 0
    if "level-1" in program_name or "level_1" in program_name:
        return 1
    if "level-2" in program_name or "level_2" in program_name:
        return 2
    return None
