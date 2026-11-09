"""Check exact staged frontend blobs and only fail changed diagnostics."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
BIOME = FRONTEND / "node_modules" / "@biomejs" / "biome" / "bin" / "biome"
SUFFIXES = {".js", ".jsx", ".ts", ".tsx", ".json", ".css"}


class ToolError(RuntimeError):
    """A required Git or Biome operation failed."""


def run_git(*args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, check=False
        )
    except OSError as exc:
        raise ToolError(f"Git execution failed during {' '.join(args)}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise ToolError(
            f"Git command failed during {' '.join(args)} "
            f"(exit code {result.returncode}){suffix}"
        )
    return result


def normalize_repo_path(value: str) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        relative = candidate.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path is outside the repository: {value}") from exc
    return relative.as_posix()


def staged_files() -> list[tuple[str, str]]:
    result = run_git(
        "diff", "--cached", "--name-status", "--find-renames", "--find-copies", "-z"
    )
    fields = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(fields) - 1:
        status = fields[index][0] if fields[index] else ""
        index += 1
        if status in {"R", "C"}:
            index += 1
        if index >= len(fields):
            break
        path = fields[index].replace("\\", "/")
        index += 1
        if status in {"A", "C", "M", "R"} and path.startswith("frontend/"):
            if Path(path).suffix.lower() in SUFFIXES:
                entries.append((status, path))
    return entries


def changed_lines(path: str) -> set[int]:
    result = run_git("diff", "--cached", "--unified=0", "--", path)
    lines: set[int] = set()
    for match in re.finditer(
        r"^@@ .* \+(\d+)(?:,(\d+))? @@",
        result.stdout.decode("utf-8", errors="replace"),
        re.MULTILINE,
    ):
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        lines.update(range(start, start + count))
    return lines


def diagnostics(path: str, content: bytes) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="pontis-biome-") as directory:
        temporary = Path(directory) / Path(path).name
        temporary.write_bytes(content)
        try:
            result = subprocess.run(
                [
                    "node",
                    str(BIOME),
                    "check",
                    str(temporary),
                    "--config-path",
                    "biome.json",
                    "--reporter=json",
                    "--max-diagnostics=none",
                    "--formatter-enabled=false",
                ],
                cwd=FRONTEND,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise ToolError(f"Biome execution failed for {path}: {exc}") from exc
        try:
            payload = json.loads(result.stdout.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise ToolError(
                f"Biome execution failed for {path} (exit code {result.returncode}){suffix}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("diagnostics"), list):
            raise ToolError(f"Biome returned unexpected JSON for {path}")
        if result.returncode and not payload["diagnostics"]:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise ToolError(
                f"Biome check failed for {path} (exit code {result.returncode})"
                f"{': ' + detail if detail else ''}"
            )
        return payload["diagnostics"]


def formatted_content(path: str, content: bytes) -> bytes:
    """Return Biome's formatted staged content without writing any file."""
    try:
        result = subprocess.run(
            [
                "node",
                str(BIOME),
                "format",
                "--stdin-file-path",
                path,
                "--config-path",
                "biome.json",
            ],
            input=content,
            cwd=FRONTEND,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ToolError(f"Biome formatter execution failed for {path}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ToolError(
            f"Biome formatter failed for {path} (exit code {result.returncode})"
            f"{': ' + detail if detail else ''}"
        )
    return result.stdout


def main() -> int:
    if not BIOME.is_file():
        print(
            "Biome is not installed. Run `corepack yarn --cwd frontend install --frozen-lockfile`.",
            file=sys.stderr,
        )
        return 1
    try:
        entries = staged_files()
        failures: list[tuple[str, str]] = []
        for _, path in entries:
            blob = run_git("show", f":{path}")
            changed = changed_lines(path)
            for diagnostic in diagnostics(path, blob.stdout):
                location = diagnostic.get("location") or {}
                line = (location.get("start") or {}).get("line")
                if diagnostic.get("category") == "parse" or line is None or line in changed:
                    failures.append((path, diagnostic.get("message", "Biome diagnostic")))
            formatted = formatted_content(path, blob.stdout)
            if formatted.replace(b"\r\n", b"\n") != blob.stdout.replace(b"\r\n", b"\n"):
                failures.append((path, "Biome formatter would change the staged content"))
    except ToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for path, message in failures:
        print(f"{path}: {message}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
