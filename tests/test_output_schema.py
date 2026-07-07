from __future__ import annotations

import unittest

from src.output_schema import TaskResult, ensure_all_styles, validate_result


class OutputSchemaTests(unittest.TestCase):
    def test_ensure_all_styles_requires_requested_keys(self) -> None:
        with self.assertRaises(ValueError):
            ensure_all_styles({"formal": "A valid formal caption."}, ("formal", "sarcastic"))

    def test_validate_result_accepts_matching_task(self) -> None:
        result = TaskResult(
            task_id="v1",
            captions={
                "formal": "A person walks through a city street carrying a bag in daylight.",
                "sarcastic": "A person walks through a city street carrying a bag, clearly the event of the century.",
            },
        )
        validate_result("v1", ("formal", "sarcastic"), result)


if __name__ == "__main__":
    unittest.main()
