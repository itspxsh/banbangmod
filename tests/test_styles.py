from __future__ import annotations

import unittest

from src.validator import validate_and_repair_captions


class StyleValidationTests(unittest.TestCase):
    def test_keeps_valid_requested_styles_only(self) -> None:
        captions = {
            "formal": "A dog runs across a grassy field while a person follows nearby in daylight.",
            "sarcastic": "A dog runs across a grassy field while a person follows nearby, clearly a deeply shocking development.",
        }

        validated = validate_and_repair_captions(
            captions,
            ("formal", "sarcastic"),
            "A dog runs across a grassy field while a person follows nearby in daylight.",
            checklist={"subjects": ["dog", "person"], "setting": "grassy field", "actions": ["runs", "follows"]},
        )

        self.assertEqual(set(validated.keys()), {"formal", "sarcastic"})

    def test_repairs_missing_tech_reference(self) -> None:
        def repair(summary: str, caption: str, style: str, checklist):
            del summary, caption, checklist
            self.assertEqual(style, "humorous_tech")
            return "A cat sits in a garden, quietly running a leaf-monitoring script under calm daylight."

        validated = validate_and_repair_captions(
            {"humorous_tech": "A cat sits in a garden and looks around in daylight."},
            ("humorous_tech",),
            "A cat sits in a garden and looks around in daylight.",
            checklist={"subjects": ["cat"], "setting": "garden", "actions": ["looks around"]},
            repair_callback=repair,
        )

        self.assertIn("script", validated["humorous_tech"].lower())

    def test_repairs_non_tech_caption_that_contains_tech_words(self) -> None:
        def repair(summary: str, caption: str, style: str, checklist):
            del summary, caption, checklist
            self.assertEqual(style, "humorous_non_tech")
            return "A cat sits in a garden like the plants just appointed it neighborhood supervisor for the afternoon."

        validated = validate_and_repair_captions(
            {"humorous_non_tech": "A cat sits in a garden while the debug script stays online."},
            ("humorous_non_tech",),
            "A cat sits in a garden and looks around in daylight.",
            checklist={"subjects": ["cat"], "setting": "garden", "actions": ["looks around"]},
            repair_callback=repair,
        )

        self.assertNotIn("script", validated["humorous_non_tech"].lower())
        self.assertNotIn("debug", validated["humorous_non_tech"].lower())

    def test_template_fallback_still_produces_all_requested_styles(self) -> None:
        validated = validate_and_repair_captions(
            {},
            ("formal", "sarcastic", "humorous_tech", "humorous_non_tech"),
            "A person walks through a city street carrying a bag in daylight.",
            checklist={"subjects": ["person"], "setting": "city street", "actions": ["walks", "carrying"]},
        )

        self.assertEqual(set(validated.keys()), {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"})
        self.assertIn("script", validated["humorous_tech"].lower())


if __name__ == "__main__":
    unittest.main()
