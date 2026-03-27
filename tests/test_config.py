from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_ai_agent.config import AppConfig


class AppConfigTests(unittest.TestCase):
    def test_config_is_complete_when_all_values_are_present(self) -> None:
        config = AppConfig(
            openai_api_key="test-openai-key",
            qdrant_url="https://example-qdrant.io",
            qdrant_api_key="test-qdrant-key",
        )

        self.assertTrue(config.is_complete)
        self.assertEqual(config.missing_fields(), [])

    def test_config_reports_missing_fields(self) -> None:
        config = AppConfig(openai_api_key="", qdrant_url="https://example-qdrant.io", qdrant_api_key="")

        self.assertFalse(config.is_complete)
        self.assertEqual(config.missing_fields(), ["OPENAI_API_KEY", "QDRANT_API_KEY"])

