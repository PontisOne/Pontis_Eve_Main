import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
BIOME = ROOT / "frontend/node_modules/@biomejs/biome/bin/biome"
CONFIG = ROOT / "frontend/biome.json"


class BiomeTailwindTests(unittest.TestCase):
    def run_biome(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.css"
            path.write_text(content, encoding="utf-8")
            return subprocess.run(
                [
                    "node",
                    str(BIOME),
                    "check",
                    str(path),
                    "--config-path",
                    str(CONFIG),
                    "--reporter=json",
                    "--max-diagnostics=none",
                ],
                cwd=CONFIG.parent,
                capture_output=True,
                check=False,
            )

    def test_project_tailwind_v3_directives_are_accepted(self):
        result = self.run_biome(
            "@tailwind base;\n"
            "@tailwind components;\n"
            "@tailwind utilities;\n"
            "@layer base {\n"
            "  body {\n"
            "    @apply bg-background text-foreground;\n"
            "  }\n"
            "}\n"
        )
        payload = json.loads(result.stdout.decode("utf-8"))
        self.assertEqual(result.returncode, 0)
        self.assertFalse(payload["diagnostics"])

    def test_unrelated_unknown_at_rule_is_still_reported(self):
        result = self.run_biome("@not-a-tailwind-rule value;\n")
        payload = json.loads(result.stdout.decode("utf-8"))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(
            any(
                diagnostic.get("category") == "lint/suspicious/noUnknownAtRules"
                for diagnostic in payload["diagnostics"]
            )
        )

    def test_frontend_entrypoint_uses_the_frontend_root_config(self):
        result = subprocess.run(
            [
                "node",
                "node_modules/@biomejs/biome/bin/biome",
                "check",
                "--config-path",
                "biome.json",
                "src/index.css",
                "--formatter-enabled=false",
                "--reporter=json",
            ],
            cwd=CONFIG.parent,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_root_staged_entrypoint_does_not_process_backend_files(self):
        result = subprocess.run(
            ["python", "scripts/biome_staged.py"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
