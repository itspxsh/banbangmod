from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.io_utils import TaskSpec, read_tasks, write_results


class FormatTests(unittest.TestCase):
    def test_read_tasks_rejects_unknown_style(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "bad",
                            "video_url": "https://example.com/video.mp4",
                            "styles": ["formal", "chaotic"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                read_tasks(path)

    def test_read_tasks_deduplicates_styles_preserving_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "v1",
                            "video_url": "https://example.com/video.mp4",
                            "styles": ["formal", "formal", "sarcastic"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            tasks = read_tasks(path)
            self.assertEqual(
                tasks,
                [TaskSpec(task_id="v1", video_url="https://example.com/video.mp4", styles=("formal", "sarcastic"))],
            )

    def test_write_results_emits_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.json"
            results = [{"task_id": "v1", "captions": {"formal": "test caption output."}}]

            write_results(path, results)

            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, results)


if __name__ == "__main__":
    unittest.main()
