from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any


CUBE_COLOR_ORDER_KEYS = (
    1,
    7,
    10,
    16,
    19,
    24,
    31,
    32,
    34,
    40,
    44,
    46,
    48,
    49,
    52,
    62,
    65,
    68,
    77,
    81,
    84,
    92,
    99,
)

START_X_RANGE = (-4.0, 2.1)
START_Y_RANGE = (-6.5, 6.4)
DEFAULT_SETUP_SEED = "interim"
DEFAULT_OBSTACLE_CLEARANCE_M = 0.45
MAX_START_SAMPLES = 10_000


@dataclass(frozen=True)
class EvaluationSetup:
    level: int
    setup_seed: str
    cube_color_order_key: int
    start_x: float
    start_y: float


def _rng_for(level: int, setup_seed: str) -> random.Random:
    return random.Random(f"hansung-menlo-eval:{setup_seed}:level-{level}")


def choose_evaluation_setup(level: int, setup_seed: str = DEFAULT_SETUP_SEED) -> EvaluationSetup:
    """Choose the shared hidden setup for a project level.

    All teams evaluated with the same level and setup_seed get the same cube
    order key and start pose. Different levels intentionally get different
    deterministic draws.
    """
    if level not in {0, 1, 2}:
        raise ValueError("level must be one of 0, 1, or 2")

    rng = _rng_for(level, setup_seed)
    return EvaluationSetup(
        level=level,
        setup_seed=setup_seed,
        cube_color_order_key=rng.choice(CUBE_COLOR_ORDER_KEYS),
        start_x=rng.uniform(*START_X_RANGE),
        start_y=rng.uniform(*START_Y_RANGE),
    )


def _box_bounds_xy(obstacle: dict[str, Any], clearance_m: float) -> tuple[float, float, float, float] | None:
    if obstacle.get("kind") != "box":
        return None
    try:
        x, y = obstacle["pose"]["position"][:2]
        sx, sy = obstacle["size"][:2]
    except (KeyError, TypeError, ValueError):
        return None
    half_x = float(sx) / 2.0 + clearance_m
    half_y = float(sy) / 2.0 + clearance_m
    return float(x) - half_x, float(x) + half_x, float(y) - half_y, float(y) + half_y


def point_is_clear_of_obstacles(
    x: float,
    y: float,
    obstacles: list[dict[str, Any]],
    *,
    clearance_m: float = DEFAULT_OBSTACLE_CLEARANCE_M,
) -> bool:
    for obstacle in obstacles:
        bounds = _box_bounds_xy(obstacle, clearance_m)
        if bounds is None:
            continue
        min_x, max_x, min_y, max_y = bounds
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return False
    return True


def choose_clear_start_xy(
    level: int,
    setup_seed: str,
    obstacles: list[dict[str, Any]],
    *,
    clearance_m: float = DEFAULT_OBSTACLE_CLEARANCE_M,
) -> tuple[float, float]:
    rng = _rng_for(level, setup_seed)
    rng.choice(CUBE_COLOR_ORDER_KEYS)

    for _attempt in range(MAX_START_SAMPLES):
        x = rng.uniform(*START_X_RANGE)
        y = rng.uniform(*START_Y_RANGE)
        if point_is_clear_of_obstacles(x, y, obstacles, clearance_m=clearance_m):
            return x, y

    raise RuntimeError(
        "Could not sample a start position clear of scene_layout obstacles. "
        f"Try lowering clearance_m below {clearance_m}."
    )


async def current_scene_id(ctx: Any) -> str | None:
    """Return the runtime scene id when the viewer exposes scene_layout."""
    try:
        layout = await ctx.state("scene_layout")
    except Exception:
        return None
    if isinstance(layout, dict):
        scene_id = layout.get("scene_id")
        return str(scene_id) if scene_id is not None else None
    scene_id = getattr(layout, "scene_id", None)
    return str(scene_id) if scene_id is not None else None


async def get_scene_layout(ctx: Any) -> dict[str, Any] | None:
    try:
        layout = await ctx.state("scene_layout")
    except Exception:
        return None
    return layout if isinstance(layout, dict) else None


async def apply_clear_start_from_layout(
    ctx: Any,
    setup: EvaluationSetup,
    *,
    clearance_m: float = DEFAULT_OBSTACLE_CLEARANCE_M,
) -> EvaluationSetup:
    layout = await get_scene_layout(ctx)
    if layout is None:
        print("scene_layout unavailable; using unfiltered sampled start position.")
        return setup

    obstacles = layout.get("obstacles", [])
    if not isinstance(obstacles, list):
        print("scene_layout obstacles unavailable; using unfiltered sampled start position.")
        return setup

    x, y = choose_clear_start_xy(
        setup.level,
        setup.setup_seed,
        obstacles,
        clearance_m=clearance_m,
    )
    return EvaluationSetup(
        level=setup.level,
        setup_seed=setup.setup_seed,
        cube_color_order_key=setup.cube_color_order_key,
        start_x=x,
        start_y=y,
    )


async def reload_current_scene(ctx: Any) -> None:
    """Reload the current scene if the runtime select_scene skill is available."""
    scene_id = await current_scene_id(ctx)
    if not scene_id:
        print("Scene id unavailable; skip select_scene reload.")
        return
    result = await ctx.invoke("select_scene", {"scene_id": scene_id}, timeout_s=30)
    status = getattr(result, "status", result)
    print(f"select_scene {scene_id!r} -> {status}")


async def go_to_start_position(ctx: Any, setup: EvaluationSetup) -> Any:
    target = {
        "kind": "pose",
        "pose": {
            "frame_id": "world",
            "position": [setup.start_x, setup.start_y, 0.0],
        },
    }
    result = await ctx.invoke("go_to", {"target": target}, timeout_s=300)
    status = getattr(result, "status", result)
    print(f"go_to start ({setup.start_x:+.2f}, {setup.start_y:+.2f}) -> {status}")
    return result


async def run(ctx: Any) -> None:
    level = int(os.environ.get("EVAL_LEVEL", "0"))
    setup_seed = os.environ.get("EVAL_SETUP_SEED", DEFAULT_SETUP_SEED)
    clearance_m = float(os.environ.get("EVAL_OBSTACLE_CLEARANCE_M", DEFAULT_OBSTACLE_CLEARANCE_M))
    setup = choose_evaluation_setup(level, setup_seed)

    setup = await apply_clear_start_from_layout(ctx, setup, clearance_m=clearance_m)

    print("=" * 60)
    print("Evaluation setup")
    print(f"level: {setup.level}")
    print(f"setup_seed: {setup.setup_seed}")
    print(f"cube_color_order_key: {setup.cube_color_order_key}")
    print(f"start_xy: ({setup.start_x:+.3f}, {setup.start_y:+.3f})")
    print(f"obstacle_clearance_m: {clearance_m:.2f}")
    print("=" * 60)

    await reload_current_scene(ctx)

    input(
        "In the viewer seed box, enter the cube_color_order_key above, "
        "apply/reset the scene, then press Enter here..."
    )

    await go_to_start_position(ctx, setup)

    print("Evaluation setup complete.")
