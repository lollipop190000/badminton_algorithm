import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_repository import LIMIT, changed_report_errors, path_errors, report_errors, run_errors
from update_manifest import digest, refresh


class RepositoryRulesTests(unittest.TestCase):
    def test_rejects_private_generated_and_oversized_files(self):
        for name in (".runtime/src/model.py", ".env", "x/.venv/bin/python", "raw/video.mp4", "x/key.pem"):
            with self.subTest(name=name):
                self.assertTrue(path_errors(name, 1))
        self.assertTrue(path_errors("01_data/results/full.npz", LIMIT + 1))
        self.assertFalse(path_errors("01_data/results/full.parquet", LIMIT))

    def test_research_needs_report_but_docs_do_not(self):
        self.assertTrue(changed_report_errors(["07_ope/code/src/evaluate.py"]))
        self.assertTrue(changed_report_errors(["07_ope/results/old.csv"]))
        self.assertFalse(changed_report_errors(["07_ope/code/DETAILS.md"]))
        self.assertFalse(changed_report_errors(["07_ope/code/src/evaluate.py", "07_ope/REPORT.md"]))

    def test_reports_require_ordered_sections_and_existing_local_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "REPORT.md"
            report.write_text("## 2. 과정\n## 1. 원인 / 하는 이유 / 개요\n## 3. 결과\n[x](missing.csv)", encoding="utf-8")
            self.assertEqual(len(report_errors(report, root)), 2)
            (root / "actual.csv").write_text("n,score\n1,0.5\n")
            report.write_text("## 1. 원인 / 하는 이유 / 개요\n## 2. 과정\n## 3. 결과\n[x](actual.csv)", encoding="utf-8")
            self.assertFalse(report_errors(report, root))

    def test_new_run_cannot_use_placeholder_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RUN.json"
            template = Path(__file__).resolve().parents[2] / "docs/templates/RUN.json"
            data = json.loads(template.read_text(encoding="utf-8"))
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertTrue(run_errors(path))
            artifact = path.parent / "metrics.json"
            artifact.write_text('{"n": 10, "score": 0.5}', encoding="utf-8")
            data.update(run_id=path.parent.name, code_commit="a" * 40, status="completed", command="python run.py test", split="fixed match split", environment={"python": "3.12", "dependencies": "requirements-ci.txt", "device": "cpu"}, metrics={"score": "synthetic example fraction"}, limitations="Synthetic validation fixture")
            data["inputs"] = [{"path_or_source": "fixture", "sha256": "b" * 64}]
            data["config"] = {"path": "fixture", "sha256": "c" * 64}
            data["outputs"] = [{"path": "metrics.json", "sha256": digest(artifact)}]
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertFalse(run_errors(path))
            artifact.write_text("changed", encoding="utf-8")
            self.assertTrue(run_errors(path))

    def test_inventory_detects_changed_bytes_without_losing_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "PUBLICATION_MANIFEST.json"
            manifest.write_text(json.dumps({"omitted_files": [{"sha256": "audited"}]}), encoding="utf-8")
            artifact = root / "result.txt"
            artifact.write_text("original", encoding="utf-8")
            with patch("update_manifest.repository_files", return_value=["PUBLICATION_MANIFEST.json", "result.txt"]):
                refresh(root)
                refresh(root, check=True)
                artifact.write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    refresh(root, check=True)
                refresh(root)
            self.assertEqual(json.loads(manifest.read_text())["omitted_files"], [{"sha256": "audited"}])


if __name__ == "__main__":
    unittest.main()
