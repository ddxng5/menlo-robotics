from __future__ import annotations

"""Menlo AI 로봇 분류 챌린지용 Level 0 프로젝트 시작 파일입니다.

이 파일은 완성된 해답이 아니라 시작 파일입니다.

지원 코드 섹션은 반복해서 작성할 필요가 없는 작은 래퍼와 자료 구조를 제공합니다.
학생 TODO 섹션은 팀의 프로젝트 설계를 직접 구현하는 부분입니다.

Level 0 규칙: scene_state, 정확한 entity ID, entity-target go_to를 사용할 수 있습니다.
핵심 과제는 고정 script가 아니라 의미 있는 LLM 보조 상위 단계 결정 loop를 구현하는 것입니다.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTracker
from menlo_runner.llm import call_llm
from menlo_runner.scene import COLOR_TO_PAD, delivered_cube_ids, held_cube_info, visible_cubes


# ---------------------------------------------------------------------------
# 지원 코드: 공통 과제 정의와 필수 LLM 결정 형식
# ---------------------------------------------------------------------------
TASK = "Find and sort cubes from the source area into their matching destination pads."

DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}

ALLOWED_NEXT_ACTIONS = {
    "search_cube",
    "navigate_to_cube",
    "pick_cube",
    "search_pad",
    "navigate_to_pad",
    "place_cube",
    "recover",
    "skip_target",
    "stop",
}

# 같은 cube target에 대한 pick/place 실패가 이 횟수를 넘으면 skip 처리한다.
MAX_ATTEMPTS_PER_TARGET = 3

# LLM 응답이 계속 invalid할 때 deterministic fallback으로 넘어가기 전 재시도 횟수.
DECISION_RETRY_LIMIT = 2

# 한 번의 LLM decision cycle에서 deterministic executor가 연속 배송할 cube 수.
DEFAULT_BATCH_LIMIT = 10
MAX_BATCH_LIMIT = 30

AGENT_SYSTEM_PROMPT = (
    "You are the high-level task supervisor for a warehouse sorting robot.\n"
    f"Task: {TASK}\n"
    "You will receive a compact JSON observation with visible cubes, held cube, "
    "delivered cube ids, color-to-pad rules, and your own memory from previous cycles.\n"
    "Reply with ONLY one JSON object and nothing else, matching this schema:\n"
    '{"next_action": "<action>", "target_color": "<color or null>", '
    '"target_entity_id": "<entity id or null>", "reason": "<short reason>", '
    '"recovery_strategy": "<optional string or null>", "batch_limit": <integer>}\n'
    f"Allowed next_action values: {sorted(ALLOWED_NEXT_ACTIONS)}.\n"
    "Your role is strategy, policy, and recovery. Do not micromanage one low-level "
    "robot action at a time. When available cubes can be delivered, choose "
    "next_action='navigate_to_cube' as permission for the deterministic executor to "
    "run a full batch loop: observe cube, navigate, pick, choose matching pad, "
    "navigate, place, verify, then repeat. Use batch_limit=10 normally. Use recover "
    "when the previous batch ended blocked, skip_target when a specific cube keeps "
    "failing, and stop only when the task is complete or no useful action remains."
)


@dataclass
class AgentDecision:
    """LLM이 반환하고 코드가 검증한 상위 단계 결정입니다."""

    next_action: str
    target_color: str | None = None
    target_entity_id: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None
    batch_limit: int = DEFAULT_BATCH_LIMIT


@dataclass
class AgentMemory:
    """observe-decide-act cycle 사이에 agent가 유지하는 상태입니다."""

    delivered_count: int = 0
    llm_decision_count: int = 0
    batch_cycles: int = 0
    held_color: str | None = None
    held_entity_id: str | None = None
    active_cube_id: str | None = None
    active_color: str | None = None
    stage: str = "need_cube"
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_cube_ids: list[str] = field(default_factory=list)
    skipped_cube_ids: list[str] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    """LLM과 실행 코드에 전달할 간결한 full-state 관찰입니다."""

    robot_status: Any
    visible_cubes: list[dict[str, Any]]
    held_cube: dict[str, str] | None
    delivered_cube_ids: list[str]
    color_to_pad: dict[str, str]
    note: str = ""


def parse_agent_decision(text: str) -> AgentDecision | None:
    """필수 구조화 LLM JSON 출력을 parse하고 validate합니다."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    next_action = data.get("next_action")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        return None

    target_color = data.get("target_color")
    if target_color is not None and not isinstance(target_color, str):
        return None

    target_entity_id = data.get("target_entity_id")
    if target_entity_id is not None and not isinstance(target_entity_id, str):
        return None

    batch_limit = data.get("batch_limit", DEFAULT_BATCH_LIMIT)
    if not isinstance(batch_limit, int):
        return None
    batch_limit = max(1, min(batch_limit, MAX_BATCH_LIMIT))

    return AgentDecision(
        next_action=next_action,
        target_color=target_color,
        target_entity_id=target_entity_id,
        reason=str(data.get("reason", "")),
        recovery_strategy=data.get("recovery_strategy"),
        batch_limit=batch_limit,
    )


def build_decision_context(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full-state 정보를 LLM에 전달하기 좋은 간결한 text context로 변환합니다."""
    return {
        "task": task,
        "visible_cubes": observation.visible_cubes,
        "held_cube": observation.held_cube,
        "delivered_cube_ids": observation.delivered_cube_ids,
        "color_to_pad": observation.color_to_pad,
        "memory": {
            "delivered_count": memory.delivered_count,
            "llm_decision_count": memory.llm_decision_count,
            "batch_cycles": memory.batch_cycles,
            "held_color": memory.held_color,
            "held_entity_id": memory.held_entity_id,
            "active_cube_id": memory.active_cube_id,
            "active_color": memory.active_color,
            "stage": memory.stage,
            "failed_attempts": memory.failed_attempts,
            "completed_cube_ids": memory.completed_cube_ids,
            "skipped_cube_ids": memory.skipped_cube_ids,
        },
        "last_result": last_result,
        "note": observation.note,
        "decision_policy": {
            "llm_role": "strategy_policy_recovery",
            "code_role": "repeat deterministic cube delivery inside each batch",
            "batch_signal": "Use next_action='navigate_to_cube' to start or continue a batch.",
            "default_batch_limit": DEFAULT_BATCH_LIMIT,
            "max_batch_limit": MAX_BATCH_LIMIT,
        },
    }


# ---------------------------------------------------------------------------
# 지원 코드: Level 0 SDK wrapper
# ---------------------------------------------------------------------------

async def get_robot_status(ctx: Any) -> Any:
    """Robot pose, motion status, neck state를 읽습니다."""
    return await ctx.state("robot_status")


async def observe_full_state(ctx: Any) -> Observation:
    """scene_state helper로 프로젝트 Level 0 관찰을 수집합니다."""
    robot_status = await get_robot_status(ctx)
    cubes = [
        {
            "entity_id": cube.entity_id,
            "color": cube.color,
            "position": cube.position,
            "distance_from_robot": round(cube.distance_from_robot, 2),
        }
        for cube in await visible_cubes(ctx)
    ]
    held = await held_cube_info(ctx)
    held_dict = {"entity_id": held[0], "color": held[1]} if held else None
    delivered = await delivered_cube_ids(ctx)
    return Observation(
        robot_status=robot_status,
        visible_cubes=cubes,
        held_cube=held_dict,
        delivered_cube_ids=delivered,
        color_to_pad=dict(COLOR_TO_PAD),
    )


async def go_to_entity(ctx: Any, entity_id: str) -> Any:
    """Level 0 entity-target navigation입니다."""
    return await ctx.invoke(
        "go_to",
        {"target": {"kind": "entity", "entity_id": entity_id}},
        timeout_s=300,
    )


async def pick_cube_by_id(ctx: Any, cube_id: str) -> Any:
    """충분히 가까이 navigation한 뒤 특정 cube entity를 pick합니다."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": cube_id}},
        timeout_s=300,
    )


async def place_on_pad_by_id(ctx: Any, pad_id: str) -> Any:
    """들고 있는 cube를 특정 pad entity에 place합니다."""
    return await ctx.invoke(
        "place_entity",
        {"target": {"kind": "entity", "entity_id": pad_id}},
        timeout_s=300,
    )


def result_summary(result: Any) -> dict[str, Any]:
    """SDK result를 log하기 쉬운 작은 dictionary로 변환합니다."""
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }


# ---------------------------------------------------------------------------
# 학생 구현: target 선택과 decision 검증을 위한 작은 helper
# ---------------------------------------------------------------------------

def choose_target_cube(observation: Observation, memory: AgentMemory) -> dict[str, Any] | None:
    """observation.visible_cubes 중 가장 가까운 available cube를 고릅니다.

    observe_world에서 completed/skipped/held cube를 이미 제외했고 scene.visible_cubes는
    거리순으로 정렬되어 있으므로, 남은 목록의 첫 항목이 가장 가까운 available cube이다.
    """
    return observation.visible_cubes[0] if observation.visible_cubes else None


def resolve_cube_target(
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any] | None:
    """decision, memory, observation 중 사용 가능한 정보로 실제 cube target을 결정합니다."""
    visible_by_id = {cube["entity_id"]: cube for cube in observation.visible_cubes}
    held = observation.held_cube
    if decision.target_entity_id:
        if decision.target_entity_id in visible_by_id:
            return visible_by_id[decision.target_entity_id]
        if held and held["entity_id"] == decision.target_entity_id:
            return held
    if memory.active_cube_id and memory.active_cube_id in visible_by_id:
        return visible_by_id[memory.active_cube_id]
    return choose_target_cube(observation, memory)


def resolve_pad_target(decision: AgentDecision, memory: AgentMemory) -> tuple[str | None, str | None]:
    """decision/memory에 남아있는 color 정보로 (color, pad_id)를 결정합니다."""
    color = decision.target_color or memory.held_color or memory.active_color
    pad_id = COLOR_TO_PAD.get(color) if color else None
    return color, pad_id


def validate_decision(decision: AgentDecision, observation: Observation, memory: AgentMemory) -> bool:
    """LLM decision이 현재 observation/memory와 모순되지 않는지 확인합니다."""
    if decision.next_action not in ALLOWED_NEXT_ACTIONS:
        return False
    if decision.target_color is not None and decision.target_color not in DESTINATION_SIGN_RULES:
        return False
    if decision.batch_limit < 1 or decision.batch_limit > MAX_BATCH_LIMIT:
        return False

    held = observation.held_cube
    visible_ids = {cube["entity_id"] for cube in observation.visible_cubes}

    if decision.next_action in ("navigate_to_cube", "pick_cube"):
        eid = decision.target_entity_id or memory.active_cube_id
        if eid is None:
            return choose_target_cube(observation, memory) is not None
        held_id = held["entity_id"] if held else None
        if eid not in visible_ids and eid != held_id:
            return False

    if decision.next_action in ("navigate_to_pad", "place_cube"):
        if held is None and memory.held_entity_id is None:
            return False
        color = decision.target_color or memory.held_color
        if color is None or color not in COLOR_TO_PAD:
            return False

    if decision.next_action == "skip_target" and not (decision.target_entity_id or memory.active_cube_id):
        return False

    return True


def fallback_decision(observation: Observation, memory: AgentMemory) -> AgentDecision:
    """LLM 응답을 계속 신뢰할 수 없을 때 사용하는 deterministic fallback decision."""
    held = observation.held_cube or (
        {"entity_id": memory.held_entity_id, "color": memory.held_color} if memory.held_entity_id else None
    )
    if held:
        if memory.stage == "pad_targeted":
            return AgentDecision(
                next_action="place_cube",
                target_color=held["color"],
                target_entity_id=held["entity_id"],
                reason="fallback: pad에 도착한 것으로 보여 place를 시도합니다.",
            )
        return AgentDecision(
            next_action="navigate_to_pad",
            target_color=held["color"],
            target_entity_id=held["entity_id"],
            reason="fallback: cube를 들고 있어 matching pad로 이동합니다.",
            batch_limit=1,
        )

    if memory.stage == "cube_targeted" and memory.active_cube_id:
        return AgentDecision(
            next_action="pick_cube",
            target_color=memory.active_color,
            target_entity_id=memory.active_cube_id,
            reason="fallback: 이미 target으로 삼은 cube를 pick 시도합니다.",
            batch_limit=1,
        )

    candidate = choose_target_cube(observation, memory)
    if candidate:
        return AgentDecision(
            next_action="navigate_to_cube",
            target_color=candidate["color"],
            target_entity_id=candidate["entity_id"],
            reason="fallback: 가장 가까운 available cube부터 batch 배송을 시작합니다.",
            batch_limit=DEFAULT_BATCH_LIMIT,
        )

    return AgentDecision(
        next_action="search_cube",
        reason="fallback: 현재 available cube가 없어 재탐색합니다.",
        batch_limit=1,
    )


# ---------------------------------------------------------------------------
# 학생 TODO: LLM decision 함수
# ---------------------------------------------------------------------------

async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    """Text LLM을 호출해 다음 상위 단계 행동을 선택하고, 실패 시 fallback으로 넘어갑니다."""
    context = build_decision_context(task, observation, memory, last_result)
    api_key = os.environ.get("TOKAMAK_API_KEY", "")
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]

    for attempt in range(1, DECISION_RETRY_LIMIT + 2):
        try:
            raw_reply = call_llm(messages, api_key=api_key)
        except Exception as exc:  # network/HTTP errors from call_llm
            memory.logs.append({"event": "llm_call_failed", "attempt": attempt, "error": str(exc)})
            continue

        decision = parse_agent_decision(raw_reply)
        if decision is not None and validate_decision(decision, observation, memory):
            memory.llm_decision_count += 1
            return decision

        memory.logs.append(
            {
                "event": "llm_invalid_decision",
                "attempt": attempt,
                "raw_reply": raw_reply[:300] if isinstance(raw_reply, str) else None,
            }
        )

    fallback = fallback_decision(observation, memory)
    memory.llm_decision_count += 1
    memory.logs.append({"event": "fallback_decision_used", "decision": fallback.__dict__})
    return fallback


# ---------------------------------------------------------------------------
# 학생 TODO: observation, execution, verification, memory
# ---------------------------------------------------------------------------

async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    """LLM과 실행 코드를 위해 현재 Level 0 관찰을 수집합니다.

    completed/skipped/held cube는 다음 target 후보에서 제외해 LLM과 fallback이 항상
    실제로 갈 수 있는 cube만 보게 합니다.
    """
    observation = await observe_full_state(ctx)
    excluded_ids = set(memory.completed_cube_ids) | set(memory.skipped_cube_ids)
    held_id = observation.held_cube["entity_id"] if observation.held_cube else None
    observation.visible_cubes = [
        cube
        for cube in observation.visible_cubes
        if cube["entity_id"] not in excluded_ids and cube["entity_id"] != held_id
    ]
    observation.note = (
        f"available_cubes={len(observation.visible_cubes)}, "
        f"completed={len(memory.completed_cube_ids)}, skipped={len(memory.skipped_cube_ids)}, "
        f"stage={memory.stage}, llm_decisions={memory.llm_decision_count}, "
        f"batch_cycles={memory.batch_cycles}"
    )
    return observation


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """검증된 LLM 결정 하나를 Level 0 robot 행동으로 변환합니다."""
    action = decision.next_action

    if action == "stop":
        return {"action": action, "status": "stopped"}

    if action == "search_cube":
        candidate = choose_target_cube(observation, memory)
        return {
            "action": action,
            "status": "found" if candidate else "not_found",
            "target_entity_id": candidate["entity_id"] if candidate else None,
            "target_color": candidate["color"] if candidate else None,
        }

    if action == "navigate_to_cube":
        batch_limit = max(1, min(decision.batch_limit, MAX_BATCH_LIMIT))
        delivered_ids: list[str] = []
        skipped_ids: list[str] = []
        attempts: list[dict[str, Any]] = []
        blocked_error: str | None = None

        for _ in range(batch_limit):
            current = await observe_full_state(ctx)
            excluded_ids = (
                set(memory.completed_cube_ids)
                | set(memory.skipped_cube_ids)
                | set(delivered_ids)
                | set(skipped_ids)
            )
            current.visible_cubes = [
                cube for cube in current.visible_cubes if cube["entity_id"] not in excluded_ids
            ]

            held = current.held_cube
            if held:
                cube_id = held["entity_id"]
                color = held["color"]
                step = {"cube_id": cube_id, "color": color, "started_held": True}
            else:
                cube = choose_target_cube(current, memory)
                if cube is None:
                    break

                cube_id = cube["entity_id"]
                color = cube["color"]
                step = {"cube_id": cube_id, "color": color, "started_held": False}

                navigate_result = await go_to_entity(ctx, cube_id)
                step["navigate_to_cube"] = result_summary(navigate_result)
                if step["navigate_to_cube"].get("error"):
                    blocked_error = step["navigate_to_cube"]["error"]
                    skipped_ids.append(cube_id)
                    attempts.append(step)
                    continue

                pick_result = await pick_cube_by_id(ctx, cube_id)
                step["pick_cube"] = result_summary(pick_result)
                after_pick = await observe_full_state(ctx)
                held_after_pick = after_pick.held_cube
                if (
                    step["pick_cube"].get("error")
                    or held_after_pick is None
                    or held_after_pick["entity_id"] != cube_id
                ):
                    blocked_error = step["pick_cube"].get("error") or "pick did not attach cube"
                    skipped_ids.append(cube_id)
                    attempts.append(step)
                    continue

                color = held_after_pick["color"]
                step["color"] = color

            pad_id = COLOR_TO_PAD.get(color)
            if pad_id is None:
                blocked_error = f"no matching pad for color {color}"
                attempts.append(step)
                break

            navigate_pad_result = await go_to_entity(ctx, pad_id)
            step["navigate_to_pad"] = result_summary(navigate_pad_result)
            step["target_pad_id"] = pad_id
            if step["navigate_to_pad"].get("error"):
                blocked_error = step["navigate_to_pad"]["error"]
                attempts.append(step)
                break

            place_result = await place_on_pad_by_id(ctx, pad_id)
            step["place_cube"] = result_summary(place_result)
            after_place = await observe_full_state(ctx)
            delivered = step["place_cube"].get("error") is None and after_place.held_cube is None
            step["delivered"] = delivered
            attempts.append(step)

            if delivered:
                delivered_ids.append(cube_id)
                continue

            blocked_error = step["place_cube"].get("error") or "place did not release cube"
            break

        if delivered_ids:
            status = "batch_complete" if len(delivered_ids) >= batch_limit else "batch_partial"
        elif blocked_error:
            status = "blocked"
        else:
            status = "no_available_cube"

        return {
            "action": "batch_delivery",
            "requested_action": action,
            "status": status,
            "batch_limit": batch_limit,
            "deliveries": len(delivered_ids),
            "delivered_cube_ids": delivered_ids,
            "skipped_cube_ids": skipped_ids,
            "attempts": attempts,
            "last_error": blocked_error,
            "error": blocked_error if not delivered_ids and blocked_error else None,
        }

    if action == "pick_cube":
        cube = resolve_cube_target(decision, observation, memory)
        if cube is None:
            return {"action": action, "status": "invalid_target", "error": "no available cube target"}
        result = await pick_cube_by_id(ctx, cube["entity_id"])
        summary = result_summary(result)
        summary.update({"action": action, "target_entity_id": cube["entity_id"], "target_color": cube["color"]})
        return summary

    if action == "search_pad":
        color, pad_id = resolve_pad_target(decision, memory)
        return {
            "action": action,
            "status": "found" if pad_id else "not_found",
            "target_color": color,
            "target_pad_id": pad_id,
        }

    if action == "navigate_to_pad":
        color, pad_id = resolve_pad_target(decision, memory)
        if pad_id is None:
            return {"action": action, "status": "invalid_target", "error": "no held/target color"}
        result = await go_to_entity(ctx, pad_id)
        summary = result_summary(result)
        summary.update({"action": action, "target_pad_id": pad_id, "target_color": color})
        return summary

    if action == "place_cube":
        color, pad_id = resolve_pad_target(decision, memory)
        if pad_id is None:
            return {"action": action, "status": "invalid_target", "error": "no held/target color"}
        result = await place_on_pad_by_id(ctx, pad_id)
        summary = result_summary(result)
        summary.update({"action": action, "target_pad_id": pad_id, "target_color": color})
        return summary

    if action == "recover":
        return {
            "action": action,
            "status": "resync",
            "recovery_strategy": decision.recovery_strategy,
        }

    if action == "skip_target":
        skip_id = decision.target_entity_id or memory.active_cube_id
        return {"action": action, "status": "ok", "skipped_entity_id": skip_id}

    return {"action": action, "status": "unhandled"}


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """마지막 action 뒤 scene_state를 다시 관찰해 held_cube/delivered 상태를 확인합니다."""
    observation = await observe_full_state(ctx)
    return {
        "decision": decision.__dict__,
        "action_result": action_result,
        "action_ok": action_result.get("error") is None,
        "held_cube": observation.held_cube,
        "delivered_cube_ids": observation.delivered_cube_ids,
        "visible_cube_ids": [cube["entity_id"] for cube in observation.visible_cubes],
    }


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    """각 cycle 뒤 held/active/delivered/failure/skip 상태를 update하고 log를 남깁니다."""
    action = decision.next_action
    action_result = verified["action_result"]
    held_cube = verified["held_cube"]
    visible_ids = set(verified["visible_cube_ids"])

    prior_held_entity_id = memory.held_entity_id

    # held 상태는 항상 scene_state 기준으로 재동기화한다 (ground truth).
    if held_cube:
        memory.held_entity_id = held_cube["entity_id"]
        memory.held_color = held_cube["color"]
    else:
        memory.held_entity_id = None
        memory.held_color = None

    if action_result.get("action") == "batch_delivery":
        for cube_id in action_result.get("delivered_cube_ids", []):
            if cube_id not in memory.completed_cube_ids:
                memory.completed_cube_ids.append(cube_id)
                memory.delivered_count += 1
            memory.failed_attempts.pop(cube_id, None)

        for cube_id in action_result.get("skipped_cube_ids", []):
            if cube_id not in memory.completed_cube_ids and cube_id not in memory.skipped_cube_ids:
                memory.skipped_cube_ids.append(cube_id)
            if cube_id not in memory.completed_cube_ids:
                memory.failed_attempts[cube_id] = memory.failed_attempts.get(cube_id, 0) + 1

        memory.batch_cycles += 1
        if memory.held_entity_id:
            memory.active_cube_id = memory.held_entity_id
            memory.active_color = memory.held_color
            memory.stage = "need_pad"
        else:
            memory.active_cube_id = None
            memory.active_color = None
            memory.stage = "need_cube"

    elif action == "navigate_to_cube":
        target_id = action_result.get("target_entity_id")
        if target_id and action_result.get("error") is None:
            memory.active_cube_id = target_id
            memory.active_color = action_result.get("target_color")
            memory.stage = "cube_targeted"

    elif action == "search_cube":
        if action_result.get("status") == "found":
            memory.active_cube_id = action_result.get("target_entity_id")
            memory.active_color = action_result.get("target_color")

    elif action == "pick_cube":
        target_id = action_result.get("target_entity_id")
        if target_id and prior_held_entity_id is None and memory.held_entity_id == target_id:
            memory.stage = "need_pad"
            memory.active_cube_id = None
            memory.failed_attempts.pop(target_id, None)
        elif target_id:
            memory.failed_attempts[target_id] = memory.failed_attempts.get(target_id, 0) + 1
            if memory.failed_attempts[target_id] >= MAX_ATTEMPTS_PER_TARGET:
                memory.skipped_cube_ids.append(target_id)
                memory.active_cube_id = None
                memory.active_color = None
                memory.stage = "need_cube"

    elif action == "navigate_to_pad":
        if action_result.get("error") is None and action_result.get("status") != "invalid_target":
            memory.stage = "pad_targeted"

    elif action == "place_cube":
        # scene_state의 delivered_cube_ids는 아직 등장하지 않은 reserve cube도
        # visible=False로 포함할 수 있어 신뢰할 수 없다. 대신 "들고 있다가 놓였는지"
        # (held -> released) 전이를 성공 판정 기준으로 사용한다.
        released = prior_held_entity_id is not None and memory.held_entity_id is None
        if released and action_result.get("error") is None:
            memory.completed_cube_ids.append(prior_held_entity_id)
            memory.delivered_count += 1
            memory.active_color = None
            memory.stage = "need_cube"
        elif prior_held_entity_id:
            memory.failed_attempts[prior_held_entity_id] = memory.failed_attempts.get(prior_held_entity_id, 0) + 1
            memory.stage = "need_pad"

    elif action == "recover":
        if memory.held_entity_id:
            memory.stage = "need_pad"
        elif memory.active_cube_id and memory.active_cube_id not in visible_ids:
            memory.failed_attempts[memory.active_cube_id] = memory.failed_attempts.get(memory.active_cube_id, 0) + 1
            if memory.failed_attempts[memory.active_cube_id] >= MAX_ATTEMPTS_PER_TARGET:
                memory.skipped_cube_ids.append(memory.active_cube_id)
                memory.active_cube_id = None
                memory.active_color = None
            memory.stage = "need_cube"
        elif memory.active_cube_id is None:
            memory.stage = "need_cube"

    elif action == "skip_target":
        skip_id = action_result.get("skipped_entity_id")
        if skip_id and skip_id not in memory.skipped_cube_ids:
            memory.skipped_cube_ids.append(skip_id)
        if skip_id == memory.active_cube_id:
            memory.active_cube_id = None
            memory.active_color = None
            memory.stage = "need_cube"

    memory.logs.append(
        {
            "cycle_action": action,
            "reason": decision.reason,
            "stage": memory.stage,
            "held_color": memory.held_color,
            "delivered_count": memory.delivered_count,
            "llm_decision_count": memory.llm_decision_count,
            "batch_cycles": memory.batch_cycles,
            "deliveries_this_cycle": action_result.get("deliveries", 0),
            "batch_status": action_result.get("status"),
            "last_error": action_result.get("last_error") or action_result.get("error"),
            "visible_cube_count": len(observation.visible_cubes),
            "skipped_cube_ids": list(memory.skipped_cube_ids),
            "action_ok": verified.get("action_ok"),
        }
    )


async def run_agent(
    ctx: Any,
    *,
    task: str = TASK,
    max_cycles: int = 10_000,
    completion: CompletionConfig | None = None,
) -> AgentMemory:
    """LLM strategy cycle마다 deterministic batch delivery를 실행합니다.

    여기서 max_cycles는 low-level robot action 수가 아니라 LLM batch decision 수입니다.
    """
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None
    tracker = CompletionTracker(completion) if completion is not None else None

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 0] LLM batch cycle {cycle}")
        if tracker is not None:
            first_cycle = tracker.started_at is None
            tracker.start_first_cycle()
            if first_cycle:
                tracker.print_start()
            reason = await tracker.stop_reason_from_scene(ctx)
            if reason is not None:
                tracker.mark_ended(reason)
                print(f"Completion target reached before cycle action: {reason}.")
                break

        observation = await observe_world(ctx, memory)
        decision = await decide_next_action(task, observation, memory, last_result)
        print("LLM decision:", decision)

        if decision.next_action == "stop":
            break

        action_result = await execute_decision(ctx, decision, observation, memory)
        verified = await verify_outcome(ctx, decision, action_result)
        update_memory(memory, observation, decision, verified)
        last_result = verified
        if tracker is not None:
            reason = await tracker.stop_reason_from_scene(ctx)
            if reason is not None:
                tracker.mark_ended(reason)
                print(f"Completion target reached after cycle action: {reason}.")
                break

    if tracker is not None:
        await tracker.print_summary_from_scene(ctx)
    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Level 0 full-state project starter 실행")
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=CompletionConfig(level=0, max_elapsed_s=600),
    )
    print("\n실행 완료.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)



