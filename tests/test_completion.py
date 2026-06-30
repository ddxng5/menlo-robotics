import asyncio
import unittest
from types import SimpleNamespace

from menlo_runner.completion import CompletionConfig, CompletionTracker, level_from_program_name


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def cube_state(
    *,
    entity_id: str,
    visible: bool,
    attached_to: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(entity_id=entity_id, visible=visible, attached_to=attached_to)


class FakeSceneContext:
    def __init__(self, entities=None) -> None:
        self.entities = entities or {
            "cube_0": cube_state(entity_id="cube_0", visible=False),
            "cube_pool_0": cube_state(entity_id="cube_pool_0", visible=False),
            "cube_1": cube_state(entity_id="cube_1", visible=True),
            "robot": SimpleNamespace(visible=True),
        }

    async def state(self, name: str):
        if name != "scene_state":
            raise AssertionError(f"unexpected state read: {name}")
        return SimpleNamespace(entities=self.entities)


class CompletionConfigTest(unittest.TestCase):
    def test_requires_at_least_one_stop_condition(self):
        with self.assertRaises(ValueError):
            CompletionConfig().validate()

    def test_rejects_non_positive_time_limit(self):
        with self.assertRaises(ValueError):
            CompletionConfig(max_elapsed_s=0).validate()

    def test_rejects_negative_cube_limit(self):
        with self.assertRaises(ValueError):
            CompletionConfig(max_delivered_cubes=-1).validate()

    def test_level_sets_delivery_points_without_score_cap(self):
        config = CompletionConfig(level=2, max_elapsed_s=600)

        self.assertEqual(config.delivery_points(), 30)


class CompletionTrackerTest(unittest.TestCase):
    def test_elapsed_time_starts_at_first_cycle(self):
        clock = FakeClock()
        tracker = CompletionTracker(CompletionConfig(max_elapsed_s=5), clock=clock)

        clock.advance(100)
        self.assertEqual(tracker.elapsed_s(), 0.0)

        tracker.start_first_cycle()
        clock.advance(3)

        self.assertEqual(tracker.elapsed_s(), 3.0)
        self.assertIsNone(tracker.stop_reason(delivered_count=0))

    def test_stops_when_time_limit_is_reached(self):
        clock = FakeClock()
        tracker = CompletionTracker(CompletionConfig(max_elapsed_s=5), clock=clock)

        tracker.start_first_cycle()
        clock.advance(5)

        self.assertEqual(tracker.stop_reason(delivered_count=0), "elapsed 5.0/5.0 seconds")

    def test_stops_when_delivery_limit_is_reached(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=2))

        tracker.start_first_cycle()

        self.assertIsNone(tracker.stop_reason(delivered_count=1))
        self.assertEqual(tracker.stop_reason(delivered_count=2), "delivered 2/2 cubes")

    def test_delivery_score_has_no_hundred_point_cap(self):
        tracker = CompletionTracker(CompletionConfig(level=2, max_elapsed_s=600))

        self.assertEqual(tracker.delivery_score(delivered_count=5), 150)

    def test_scene_delivered_count_ignores_initial_hidden_pool_cubes(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=1))
        ctx = FakeSceneContext()

        delivered_count = asyncio.run(tracker.scene_delivered_count(ctx))

        self.assertEqual(delivered_count, 0)

    def test_scene_delivered_count_tracks_active_cube_that_disappears(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=1))
        entities = {
            "cube_0": cube_state(entity_id="cube_0", visible=True),
            "cube_pool_0": cube_state(entity_id="cube_pool_0", visible=False),
            "robot": SimpleNamespace(visible=True),
        }
        ctx = FakeSceneContext(entities)

        self.assertEqual(asyncio.run(tracker.scene_delivered_count(ctx)), 0)
        entities["cube_0"] = cube_state(entity_id="cube_0", visible=False)

        self.assertEqual(asyncio.run(tracker.scene_delivered_count(ctx)), 1)

    def test_scene_delivered_count_tracks_pool_cube_after_it_becomes_active(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=1))
        entities = {
            "cube_pool_0": cube_state(entity_id="cube_pool_0", visible=False),
            "robot": SimpleNamespace(visible=True),
        }
        ctx = FakeSceneContext(entities)

        self.assertEqual(asyncio.run(tracker.scene_delivered_count(ctx)), 0)
        entities["cube_pool_0"] = cube_state(entity_id="cube_pool_0", visible=True)
        self.assertEqual(asyncio.run(tracker.scene_delivered_count(ctx)), 0)
        entities["cube_pool_0"] = cube_state(entity_id="cube_pool_0", visible=False)

        self.assertEqual(asyncio.run(tracker.scene_delivered_count(ctx)), 1)

    def test_stop_reason_from_scene_uses_tracked_delivered_count(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=2))
        entities = {
            "cube_0": cube_state(entity_id="cube_0", visible=True),
            "cube_1": cube_state(entity_id="cube_1", visible=True),
            "cube_pool_0": cube_state(entity_id="cube_pool_0", visible=False),
            "robot": SimpleNamespace(visible=True),
        }
        ctx = FakeSceneContext(entities)

        self.assertIsNone(asyncio.run(tracker.stop_reason_from_scene(ctx)))
        entities["cube_0"] = cube_state(entity_id="cube_0", visible=False)
        self.assertIsNone(asyncio.run(tracker.stop_reason_from_scene(ctx)))
        entities["cube_1"] = cube_state(entity_id="cube_1", visible=False)
        reason = asyncio.run(tracker.stop_reason_from_scene(ctx))

        self.assertEqual(reason, "delivered 2/2 cubes")


class ProgramLevelTest(unittest.TestCase):
    def test_infers_level_from_program_name(self):
        self.assertEqual(level_from_program_name("level-0-starter"), 0)
        self.assertEqual(level_from_program_name("level-1-starter-ko"), 1)
        self.assertEqual(level_from_program_name("menlo_runner.programs.project.en.level_2_starter"), 2)
        self.assertIsNone(level_from_program_name("student-program"))


if __name__ == "__main__":
    unittest.main()
