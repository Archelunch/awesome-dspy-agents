from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache")

from typer.testing import CliRunner

from awesome_dspy_agents.cli import app


class SessionSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runner = CliRunner()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_session(self, data: dict) -> Path:
        path = Path(self.temp_dir.name) / "session.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_replay_rejects_schema_one(self) -> None:
        path = self.write_session({"schema": 1})

        result = self.runner.invoke(app, ["replay", str(path)])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("Only session schema 2 is supported", result.output)

    def test_replay_accepts_schema_two(self) -> None:
        path = self.write_session(
            {
                "schema": 2,
                "pattern": "debate",
                "topic": "Topic",
                "tool_events_by_iter": {},
                "result": {
                    "final_answer": "Answer",
                    "justification": "Because",
                    "history": [],
                },
            }
        )

        result = self.runner.invoke(app, ["replay", str(path)])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Answer", result.output)


if __name__ == "__main__":
    unittest.main()
