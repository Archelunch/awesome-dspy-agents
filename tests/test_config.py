from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DSPY_CACHEDIR", f"{tempfile.gettempdir()}/dspy-agents-test-cache")

from awesome_dspy_agents.config import load_config, merge_config_layers


class ConfigurationTests(unittest.TestCase):
    def test_openrouter_deepseek_profile_configures_every_pattern_role(self) -> None:
        profile = (
            Path(__file__).parents[1]
            / "examples/configs/openrouter-deepseek-v4-flash.yaml"
        )

        config = load_config(str(profile))

        self.assertEqual(config.default_lm.provider, "openai")
        self.assertEqual(config.default_lm.model, "deepseek/deepseek-v4-flash")
        self.assertEqual(config.default_lm.api_base, "https://openrouter.ai/api/v1")
        self.assertEqual(
            set(config.agents),
            {"affirmative", "negative", "addition", "subtraction"},
        )
        self.assertTrue(
            all(agent.module_type == "react" for agent in config.agents.values())
        )

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
