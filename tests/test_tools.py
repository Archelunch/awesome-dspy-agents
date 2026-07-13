from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache")

from awesome_dspy_agents.tools.registry import (
    FileAccessPolicy,
    ToolExecutor,
    default_catalog,
    math_eval,
)


class ToolSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_file_access_is_denied_when_no_roots_are_configured(self) -> None:
        executor = ToolExecutor(default_catalog, FileAccessPolicy())

        result = executor.get("write_file")(str(self.root / "blocked.txt"), "secret")

        self.assertIn("access denied", result)
        self.assertFalse((self.root / "blocked.txt").exists())

    def test_file_access_is_limited_to_explicit_roots(self) -> None:
        executor = ToolExecutor(
            default_catalog, FileAccessPolicy.from_paths([self.root])
        )

        result = executor.get("write_file")(str(self.root / "allowed.txt"), "ok")

        self.assertEqual(Path(result).read_text(encoding="utf-8"), "ok")

    def test_symlink_cannot_escape_an_allowed_root(self) -> None:
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: outside.rmdir())
        (self.root / "escape").symlink_to(outside, target_is_directory=True)
        policy = FileAccessPolicy.from_paths([self.root])

        self.assertFalse(policy.allows(self.root / "escape" / "file.txt"))

    def test_math_eval_accepts_arithmetic_but_rejects_python(self) -> None:
        self.assertEqual(math_eval("2 + 3 * 4"), "14")
        self.assertIn("only arithmetic", math_eval("__import__('os').getcwd()"))

    def test_math_eval_rejects_oversized_intermediate_results(self) -> None:
        result = math_eval("((9**100)**100)**100")

        self.assertEqual(result, "error: arithmetic result is too large")

    def test_executors_do_not_share_file_permissions(self) -> None:
        allowed = ToolExecutor(
            default_catalog, FileAccessPolicy.from_paths([self.root])
        )
        denied = ToolExecutor(default_catalog, FileAccessPolicy())

        self.assertNotIn(
            "error", allowed.get("write_file")(str(self.root / "one.txt"), "ok")
        )
        self.assertIn(
            "access denied",
            denied.get("write_file")(str(self.root / "two.txt"), "blocked"),
        )

    def test_tool_listener_failure_does_not_abort_execution(self) -> None:
        def broken_listener(_event):
            raise RuntimeError("collector unavailable")

        executor = ToolExecutor(
            default_catalog, FileAccessPolicy(), listener=broken_listener
        )

        self.assertEqual(executor.get("word_count")("one two"), "2")


if __name__ == "__main__":
    unittest.main()
