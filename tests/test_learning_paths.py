from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE_MANIFEST = ROOT / "learning_paths" / "manifest.json"
PIPELINE_MANIFEST = ROOT / "pipelines" / "manifest.json"
RUNNER = ROOT / "scripts" / "run_learning_path.py"


class LearningPathContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.route_data = json.loads(ROUTE_MANIFEST.read_text(encoding="utf-8"))
        cls.pipeline_data = json.loads(PIPELINE_MANIFEST.read_text(encoding="utf-8"))

    def test_exactly_seven_unique_routes_cover_every_pipeline(self) -> None:
        routes = self.route_data["routes"]
        route_ids = [route["id"] for route in routes]
        covered = {pipeline for route in routes for pipeline in route["pipelines"]}
        registered = {pipeline["id"] for pipeline in self.pipeline_data["pipelines"]}

        self.assertEqual(7, len(routes))
        self.assertEqual(len(route_ids), len(set(route_ids)))
        self.assertEqual(registered, covered)

    def test_routes_are_bilingual_and_reference_real_foundations(self) -> None:
        bilingual_fields = (
            ("title", "title_zh"),
            ("question", "question_zh"),
            ("deliverable", "deliverable_zh"),
            ("promotion_gate", "promotion_gate_zh"),
            ("boundary", "boundary_zh"),
        )
        for route in self.route_data["routes"]:
            for english, chinese in bilingual_fields:
                self.assertTrue(route[english].strip(), f"{route['id']}.{english}")
                self.assertTrue(route[chinese].strip(), f"{route['id']}.{chinese}")
            self.assertTrue(route["metrics"])
            for relative in route["foundations"]:
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_bilingual_guides_expose_every_route_anchor(self) -> None:
        guides = (
            ROOT / "docs" / "learning-paths" / "README.md",
            ROOT / "docs" / "learning-paths" / "README_CN.md",
        )
        for guide in guides:
            text = guide.read_text(encoding="utf-8")
            for route in self.route_data["routes"]:
                self.assertIn(f'<a id="{route["id"]}"></a>', text)

    def test_route_runner_validates_and_localizes(self) -> None:
        validated = subprocess.run(
            [sys.executable, str(RUNNER), "--validate"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(0, validated.returncode, validated.stderr)
        self.assertIn("7 research routes", validated.stdout)

        shown = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--show",
                "dexterity-teleoperation",
                "--lang",
                "zh",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(0, shown.returncode, shown.stderr)
        self.assertIn("灵巧操作、重定向与遥操作", shown.stdout)
        self.assertIn("证据边界", shown.stdout)


if __name__ == "__main__":
    unittest.main()
