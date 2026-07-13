from __future__ import annotations

import unittest

from awesome_dspy_agents.patterns.deliberation import (
    DeliberationTrajectory,
    HistoryView,
)


class DeliberationTrajectoryTests(unittest.TestCase):
    def test_record_returns_a_new_trajectory_with_causal_events(self) -> None:
        empty = DeliberationTrajectory()

        proposed = empty.record(
            round_index=1,
            role="proposer-1",
            kind="proposal",
            content="Initial answer",
        )
        revised = proposed.record(
            round_index=1,
            role="proposer-1",
            kind="revision",
            content="Revised answer",
            parent_ids=("event-1",),
        )

        self.assertEqual(empty.events, ())
        self.assertEqual(proposed.events[0].event_id, "event-1")
        self.assertEqual(revised.events[1].parent_ids, ("event-1",))

    def test_render_can_select_and_anonymize_messages(self) -> None:
        trajectory = DeliberationTrajectory()
        trajectory = trajectory.record(
            round_index=1,
            role="proposer-1",
            kind="proposal",
            content="First answer",
            reasoning="private chain",
        )
        trajectory = trajectory.record(
            round_index=1,
            role="proposer-2",
            kind="proposal",
            content="Second answer",
        )

        rendered = trajectory.render(
            HistoryView(event_ids=("event-2",), anonymize_roles=True)
        )

        self.assertIn("Agent A", rendered)
        self.assertIn("Second answer", rendered)
        self.assertNotIn("proposer-2", rendered)
        self.assertNotIn("First answer", rendered)
        self.assertNotIn("private chain", rendered)


if __name__ == "__main__":
    unittest.main()
