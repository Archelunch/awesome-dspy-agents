from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from awesome_dspy_agents.patterns.interface import discover_patterns


class PatternDiscoveryTests(unittest.TestCase):
    def make_pattern_directory(self, name: str) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        directory = root / name
        directory.mkdir()
        (directory / "pattern.py").write_text("# marker\n", encoding="utf-8")
        return root

    def test_import_failure_is_returned_as_a_typed_issue(self) -> None:
        root = self.make_pattern_directory("broken")

        with patch(
            "builtins.__import__", side_effect=ImportError("missing dependency")
        ):
            catalog = discover_patterns(root)

        self.assertEqual(catalog.patterns, {})
        self.assertEqual(catalog.issues[0].pattern, "broken")
        self.assertIn("missing dependency", catalog.issues[0].message)

    def test_declared_name_must_match_directory(self) -> None:
        root = self.make_pattern_directory("folder_name")
        module = SimpleNamespace(
            get_pattern=lambda: SimpleNamespace(name="different_name")
        )

        with patch("builtins.__import__", return_value=module):
            catalog = discover_patterns(root)

        self.assertEqual(catalog.patterns, {})
        self.assertIn("does not match", catalog.issues[0].message)


if __name__ == "__main__":
    unittest.main()
