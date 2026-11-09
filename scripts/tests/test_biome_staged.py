import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.biome_staged as checker


class BiomeStagedTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Quality Tests"], cwd=self.root, check=True)
        (self.root / "frontend").mkdir()
        self.original = (checker.ROOT, checker.FRONTEND)
        checker.ROOT = self.root
        checker.FRONTEND = Path(__file__).parents[2] / "frontend"

    def tearDown(self):
        checker.ROOT, checker.FRONTEND = self.original
        self.directory.cleanup()

    def write_and_commit(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", relative], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=self.root, check=True)

    def stage(self, relative, content):
        path = self.root / relative
        path.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", relative], cwd=self.root, check=True)

    def run_checker(self):
        with mock.patch.object(checker, "BIOME", checker.FRONTEND / "node_modules/@biomejs/biome/bin/biome"):
            return checker.main()

    def test_empty_staged_input_passes(self):
        self.assertEqual(self.run_checker(), 0)

    def test_existing_diagnostic_on_untouched_line_does_not_block(self):
        path = "frontend/legacy.jsx"
        self.write_and_commit(path, "export default () => <button onClick={() => null}>x</button>;\n")
        self.stage(path, "export default () => <button onClick={() => null}>x</button>;\n// unrelated staged edit\n")
        self.assertEqual(self.run_checker(), 0)

    def test_new_button_diagnostic_blocks(self):
        self.stage("frontend/new.jsx", "export default () => <button onClick={() => null}>x</button>;\n")
        self.assertEqual(self.run_checker(), 1)

    def test_formatting_violation_blocks(self):
        self.stage("frontend/format.jsx", "export const value={answer:42};\n")
        self.assertEqual(self.run_checker(), 1)

    def test_properly_formatted_staged_file_passes(self):
        self.stage("frontend/format.jsx", "export const value = { answer: 42 };\n")
        self.assertEqual(self.run_checker(), 0)

    def test_partial_staging_uses_index_content(self):
        path = "frontend/partial.jsx"
        self.write_and_commit(path, "export default () => <button type=\"button\">x</button>;\n")
        self.stage(path, "export default () => <button type=\"button\">x</button>;\n")
        (self.root / path).write_text("export default () => <button onClick={() => null}>x</button>;\n", encoding="utf-8")
        self.assertEqual(self.run_checker(), 0)

    def test_formatting_check_does_not_modify_worktree_or_index(self):
        path = "frontend/format.jsx"
        content = "export const value={answer:42};\n"
        self.stage(path, content)
        before_worktree = (self.root / path).read_bytes()
        before_index = subprocess.run(
            ["git", "show", f":{path}"], cwd=self.root, capture_output=True, check=True
        ).stdout
        self.assertEqual(self.run_checker(), 1)
        self.assertEqual((self.root / path).read_bytes(), before_worktree)
        self.assertEqual(
            subprocess.run(
                ["git", "show", f":{path}"], cwd=self.root, capture_output=True, check=True
            ).stdout,
            before_index,
        )

    def test_deleted_frontend_file_is_ignored(self):
        path = "frontend/deleted.jsx"
        self.write_and_commit(path, "export default () => <button onClick={() => null}>x</button>;\n")
        (self.root / path).unlink()
        subprocess.run(["git", "add", "-u", path], cwd=self.root, check=True)
        self.assertEqual(self.run_checker(), 0)

    def test_no_staged_files_is_distinct_from_git_failure(self):
        self.assertEqual(checker.staged_files(), [])
        with mock.patch.object(checker, "run_git", side_effect=checker.ToolError("Git failed")):
            with self.assertRaises(checker.ToolError):
                checker.staged_files()

    def test_git_show_failure_does_not_skip_staged_file(self):
        self.stage("frontend/failure.jsx", "export default 1;\n")
        with mock.patch.object(checker, "run_git", side_effect=[
            subprocess.CompletedProcess([], 0, b"M\0frontend/failure.jsx\0", b""),
            checker.ToolError("show failed"),
        ]):
            self.assertEqual(checker.main(), 1)

    def test_missing_biome_is_a_failure(self):
        with mock.patch.object(checker, "BIOME", self.root / "missing-biome"):
            self.assertEqual(checker.main(), 1)

    def test_biome_nonzero_exit_is_reported(self):
        with mock.patch.object(
            checker.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 2, b"", b"syntax failure"),
        ):
            with self.assertRaisesRegex(checker.ToolError, "Biome execution failed"):
                checker.diagnostics("frontend/example.jsx", b"export default 1;\n")

    def test_invalid_biome_json_is_reported(self):
        with self.assertRaises(checker.ToolError):
            with mock.patch.object(checker.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"not-json", b"")):
                checker.diagnostics("frontend/example.jsx", b"export default 1;\n")

    def test_paths_and_nul_delimited_names_are_safe(self):
        path = "frontend/name with\ttab.jsx"
        output = f"A\0{path}\0".encode()
        with mock.patch.object(
            checker,
            "run_git",
            return_value=subprocess.CompletedProcess([], 0, output, b""),
        ):
            self.assertEqual(checker.staged_files(), [("A", path)])
        self.assertEqual(checker.normalize_repo_path(str(self.root / "frontend/name with tab.jsx")), "frontend/name with tab.jsx")
        self.assertEqual(checker.normalize_repo_path("frontend/name with tab.jsx"), "frontend/name with tab.jsx")


if __name__ == "__main__":
    unittest.main()
