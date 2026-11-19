import json
import os
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from contextlib import nullcontext

from scripts import backend_quality


class RuffBaselineTests(unittest.TestCase):
    @patch.object(
        backend_quality,
        "subprocess",
        **{
            "run.return_value": subprocess.CompletedProcess(
                [], 0, stdout="not json", stderr=""
            )
        },
    )
    def test_ruff_lint_counts_reports_invalid_json(self, mock_subprocess):
        with patch("sys.stderr") as stderr:
            counts, returncode = backend_quality.ruff_lint_counts([])

        self.assertFalse(counts)
        self.assertEqual(returncode, 1)
        self.assertIn(
            "Unable to parse Ruff JSON output", stderr.write.call_args_list[0].args[0]
        )
        mock_subprocess.run.assert_called_once()

    def test_load_baseline_reports_missing_file(self):
        with patch.object(
            backend_quality, "BASELINE", Path("missing-ruff-baseline.json")
        ):
            with self.assertRaisesRegex(ValueError, "Unable to load Ruff baseline"):
                backend_quality.load_baseline()

    def test_load_baseline_reports_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as stream:
            stream.write("not json")
            path = Path(stream.name)
        try:
            with patch.object(backend_quality, "BASELINE", path):
                with self.assertRaisesRegex(ValueError, "Invalid Ruff baseline"):
                    backend_quality.load_baseline()
        finally:
            path.unlink(missing_ok=True)

    @patch.object(
        backend_quality, "load_baseline", side_effect=ValueError("baseline test error")
    )
    @patch.object(backend_quality, "ruff_lint_counts", return_value=(Counter(), 0))
    def test_run_ruff_lint_reports_baseline_error(self, mock_counts, mock_baseline):
        with patch("sys.stderr") as stderr:
            result = backend_quality.run_ruff_lint([])

        self.assertEqual(result, 1)
        self.assertIn("baseline test error", stderr.write.call_args_list[0].args[0])
        mock_counts.assert_called_once_with([])
        mock_baseline.assert_called_once_with()

    def test_existing_violations_are_allowed(self):
        baseline = Counter({"app/example.py|E501|value = 1": 2})
        self.assertFalse(backend_quality.new_lint_violations(baseline, baseline))

    def test_new_violation_fails(self):
        baseline = Counter({"app/example.py|E501|value = 1": 2})
        current = Counter({"app/example.py|E501|value = 1": 3})
        self.assertEqual(
            backend_quality.new_lint_violations(current, baseline),
            Counter({"app/example.py|E501|value = 1": 1}),
        )

    def test_removing_a_violation_passes(self):
        baseline = Counter({"app/example.py|E501|value = 1": 2})
        current = Counter({"app/example.py|E501|value = 1": 1})
        self.assertFalse(backend_quality.new_lint_violations(current, baseline))

    def test_line_movement_does_not_change_rule_counts(self):
        baseline = Counter(
            {"app/example.py|E501|value = 1": 2, "app/example.py|I001|import x": 1}
        )
        moved = Counter(
            {"app/example.py|E501|value = 1": 2, "app/example.py|I001|import x": 1}
        )
        self.assertFalse(backend_quality.new_lint_violations(moved, baseline))

    def test_baseline_is_reviewable_json(self):
        baseline = Path(backend_quality.BASELINE)
        data = json.loads(baseline.read_text(encoding="utf-8"))
        self.assertEqual(data["strategy"], "file-rule-source-fingerprint")
        self.assertIn("violations", data)

    def test_format_check_does_not_modify_a_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=backend_quality.BACKEND,
            delete=False,
            encoding="utf-8",
        ) as stream:
            stream.write("value = 1\n")
            path = Path(stream.name)
        try:
            before = path.read_bytes()
            self.assertEqual(backend_quality.run_ruff([path.as_posix()]), 1)
            self.assertEqual(path.read_bytes(), before)
        finally:
            path.unlink(missing_ok=True)

    @patch.object(backend_quality, "diagnostic_counts", return_value=Counter())
    @patch.object(backend_quality, "format_check", return_value=True)
    @patch.object(backend_quality, "main_worktree")
    def test_run_ruff_forwards_only_supplied_filenames(self, mock_worktree, mock_format, mock_counts):
        supplied = ["backend/app/example.py", "backend/tests/test_example.py"]
        mock_worktree.return_value = nullcontext(Path(backend_quality.ROOT))

        self.assertEqual(backend_quality.run_ruff(supplied), 0)
        self.assertEqual(mock_counts.call_count, 1)
        self.assertEqual(mock_format.call_count, 2)

    def test_diagnostic_fingerprint_ignores_line_numbers(self):
        first = backend_quality.diagnostic_fingerprint("backend/server.py", "E501", "Line too long")
        second = backend_quality.diagnostic_fingerprint("server.py", "E501", "  Line   too long  ")
        self.assertEqual(first, second)

    def test_new_diagnostic_comparison_handles_fixed_and_moved(self):
        baseline = Counter({"server.py|E501|Line too long": 1})
        current = Counter({"server.py|E501|Line too long": 1, "server.py|B007|Unused": 1})
        self.assertEqual(
            backend_quality.compare_diagnostics(current, baseline),
            Counter({"server.py|B007|Unused": 1}),
        )
        self.assertFalse(backend_quality.compare_diagnostics(Counter(), baseline))

    def test_formatting_comparison_cases(self):
        self.assertFalse(backend_quality.formatting_is_new(False, False, True))
        self.assertTrue(backend_quality.formatting_is_new(False, True, True))
        self.assertFalse(backend_quality.formatting_is_new(True, False, True))
        self.assertTrue(backend_quality.formatting_is_new(False, True, False))


class MainWorktreeEnvironmentTests(unittest.TestCase):
    @patch.object(
        backend_quality.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    )
    def test_worktree_operations_ignore_pre_commit_index_override(self, mock_run):
        with patch.dict(
            os.environ,
            {
                "GIT_INDEX_FILE": "C:/pre-commit/index",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "credential.helper",
                "GIT_CONFIG_VALUE_0": "manager",
            },
            clear=False,
        ):
            with backend_quality.main_worktree():
                pass

        self.assertEqual(mock_run.call_count, 2)
        create_environment = mock_run.call_args_list[0].kwargs["env"]
        remove_environment = mock_run.call_args_list[1].kwargs["env"]
        self.assertNotIn("GIT_INDEX_FILE", create_environment)
        self.assertNotIn("GIT_INDEX_FILE", remove_environment)
        self.assertEqual(create_environment["GIT_CONFIG_COUNT"], "1")
        self.assertEqual(remove_environment["GIT_CONFIG_VALUE_0"], "manager")


if __name__ == "__main__":
    unittest.main()
