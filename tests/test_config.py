from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from awesome_dspy_agents.config import load_config, merge_config_layers


class ConfigurationTests(unittest.TestCase):
    def test_nested_overrides_preserve_sibling_values(self) -> None:
        base = {"debate": {"max_iterations": 3, "adaptive_break": True}}

        merged = merge_config_layers(base, {"debate": {"max_iterations": 5}})

        self.assertEqual(
            merged,
            {"debate": {"max_iterations": 5, "adaptive_break": True}},
        )
        self.assertEqual(base["debate"]["max_iterations"], 3)

    def test_lists_replace_instead_of_merging(self) -> None:
        merged = merge_config_layers(
            {"agents": {"one": {"tools": ["a", "b"]}}},
            {"agents": {"one": {"tools": ["c"]}}},
        )

        self.assertEqual(merged["agents"]["one"]["tools"], ["c"])

    def test_unknown_configuration_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("debate:\n  max_iteratons: 3\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "max_iteratons"):
                load_config(str(path))

    def test_invalid_override_is_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("debate:\n  max_iterations: 3\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "greater than or equal to 1"):
                load_config(str(path), {"debate": {"max_iterations": 0}})


if __name__ == "__main__":
    unittest.main()
