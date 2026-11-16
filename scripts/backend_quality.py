import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
BASELINE = BACKEND / "ruff-baseline.json"


def configure_console_encoding() -> None:
    """Keep diagnostics printable on Windows consoles with legacy code pages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


# Selects the Python interpreter for running development tools.
# It prefers the backend virtual environment and falls back to the
# currently running Python interpreter if the backend environment is unavailable.
def tool_python() -> str:
    """Prefer the backend virtual environment for developer tools."""
    candidates = (
        BACKEND / ".venv" / "Scripts" / "python.exe",
        BACKEND / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


# Executes the given command from the repository root and returns its exit code.
# A zero exit code generally indicates success, while a nonzero code indicates
# a failure or another result that the calling function must handle.
def run_command(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT, check=False).returncode


# Converts a file path into a path relative to the backend directory,
# using forward slashes to keep the path format consistent across operating systems.
def relative_backend_path(filename: str) -> str:
    return Path(filename).resolve().relative_to(BACKEND.resolve()).as_posix()


# Creates a stable fingerprint for a Ruff violation using its backend-relative
# file path, rule code, and normalized source line. This helps compare current
# violations with the existing baseline.
def violation_key(item: dict) -> str:
    path = relative_backend_path(item["filename"])
    row = item["location"]["row"]
    source = Path(item["filename"]).read_text(encoding="utf-8").splitlines()[row - 1]
    normalized_source = " ".join(source.split())
    return f"{path}|{item['code']}|{normalized_source}"


def diagnostic_fingerprint(path: str, code: str, message: str) -> str:
    """Create a stable diagnostic identity that deliberately ignores line numbers."""
    normalized_path = Path(path).as_posix().removeprefix("backend/")
    normalized_message = " ".join(message.split())
    return f"{normalized_path}|{code}|{normalized_message}"


def relative_diagnostic_path(filename: str, cwd: Path) -> str:
    path = Path(filename)
    try:
        return path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        parts = path.resolve().parts
        if "backend" in parts:
            return Path(*parts[parts.index("backend") + 1 :]).as_posix()
        return path.name


def compare_diagnostics(current: Counter[str], baseline: Counter[str]) -> Counter[str]:
    """Return only diagnostics newly introduced relative to the main snapshot."""
    return Counter(
        {
            key: count - baseline[key]
            for key, count in current.items()
            if count > baseline[key]
        }
    )


def main_file(path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"main:{path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        if "does not exist in" in result.stderr or "exists on disk, but not in" in result.stderr:
            return ""
        raise RuntimeError(f"Unable to read main baseline for {path}: {result.stderr.strip()}")
    return result.stdout


def nested_git_environment() -> dict[str, str]:
    """Use the real repository index for Git operations nested in a hook.

    pre-commit sets GIT_INDEX_FILE to an isolated index while running hooks.
    That is useful for staged-file checks, but it must not leak into worktree
    management: Git may otherwise try to create the worktree's index lock next
    to the temporary hook index.  Keep the rest of the environment intact so
    Git configuration, authentication helpers, PATH, and platform settings
    continue to work normally.
    """
    environment = os.environ.copy()
    environment.pop("GIT_INDEX_FILE", None)
    return environment


@contextmanager
def main_worktree():
    temporary = tempfile.mkdtemp(prefix="backend-quality-main-")
    try:
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", temporary, "main"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, env=nested_git_environment(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Unable to create the main baseline worktree. Ensure the local "
                f"main branch exists and is readable. Git reported: {result.stderr.strip()}"
            )
        yield Path(temporary)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", temporary],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, env=nested_git_environment(),
        )


@contextmanager
def baseline_paths(files: list[str]):
    temporary = tempfile.TemporaryDirectory(prefix="backend-quality-", dir=BACKEND)
    paths = []
    try:
        for filename in files:
            relative = Path(filename).resolve().relative_to(BACKEND.resolve())
            target = Path(temporary.name) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = main_file(filename)
            if not source:
                continue
            target.write_text(source, encoding="utf-8")
            paths.append(str(target))
        yield paths
    finally:
        temporary.cleanup()


def ruff_lint_counts(files: list[str]) -> tuple[Counter[str], int]:
    result = subprocess.run(
        [
            tool_python(),
            "-m",
            "ruff",
            "check",
            "--output-format",
            "json",
            "--config",
            str(BACKEND / "pyproject.toml"),
            *files,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        print(result.stderr, file=sys.stderr, end="")
        return Counter(), result.returncode
    try:
        diagnostics = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as error:
        print(f"Unable to parse Ruff JSON output: {error}", file=sys.stderr)
        return Counter(), 1
    return Counter(violation_key(item) for item in diagnostics), 0


def load_baseline() -> Counter[str]:
    try:
        with BASELINE.open(encoding="utf-8") as stream:
            data = json.load(stream)
        violations = data["violations"]
        return Counter({str(key): int(value) for key, value in violations.items()})
    except (FileNotFoundError, OSError) as error:
        raise ValueError(
            f"Unable to load Ruff baseline at {BASELINE}: {error}"
        ) from error
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Ruff baseline at {BASELINE}: {error}") from error


def new_lint_violations(current: Counter[str], baseline: Counter[str]) -> Counter[str]:
    return compare_diagnostics(current, baseline)


def run_ruff_lint(files: list[str]) -> int:
    current, returncode = ruff_lint_counts(files)
    if returncode != 0:
        return returncode
    try:
        baseline = load_baseline()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    new_violations = new_lint_violations(current, baseline)
    if new_violations:
        print("New Ruff violations detected:")
        for key, count in sorted(new_violations.items()):
            print(f"  {key}: {count}")
        return 1
    print("Ruff lint passed against the checked-in baseline.")
    return 0


def update_baseline() -> int:
    files = [str(BACKEND)]
    counts, returncode = ruff_lint_counts(files)
    if returncode != 0:
        return returncode
    data = {
        "version": "0.14.3",
        "strategy": "file-rule-source-fingerprint",
        "violations": dict(sorted(counts.items())),
    }
    BASELINE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Updated Ruff baseline at {BASELINE}.")
    return 0


def staged_python_files() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--diff-filter=ACMR",
            "--",
            "backend",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = []
    backend_root = BACKEND.resolve()
    for name in result.stdout.split("\0"):
        if not name:
            continue
        path = Path(name)
        absolute_path = (ROOT / path).resolve()
        if (
            path.suffix == ".py"
            and backend_root in absolute_path.parents
            and absolute_path.is_file()
        ):
            files.append(path.as_posix())
    return files


def diagnostic_counts(tool: str, files: list[str], cwd: Path) -> Counter[str]:
    if tool == "ruff":
        command = [tool_python(), "-m", "ruff", "check", "--output-format", "json", "--config", str(cwd / "backend" / "pyproject.toml"), *files]
    else:
        command = [tool_python(), "-m", "mypy", "--no-pretty", "--config-file", str(cwd / "backend" / "mypy.ini"), *files]
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"{tool} failed: {result.stderr.strip()}")
    diagnostics = Counter()
    if tool == "ruff":
        try:
            items = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Unable to parse Ruff output: {error}") from error
        for item in items:
            path = relative_diagnostic_path(item["filename"], cwd)
            diagnostics[diagnostic_fingerprint(path, item["code"], item["message"])] += 1
    else:
        pattern = re.compile(r"^(.*?):\d+: error: (.*?)(?: \[(\w+)\])?$")
        for line in result.stdout.splitlines():
            match = pattern.match(line.strip())
            if match:
                path = relative_diagnostic_path(match.group(1), cwd)
                diagnostics[diagnostic_fingerprint(path, match.group(3) or "mypy", match.group(2))] += 1
    return diagnostics


def format_check(files: list[str], cwd: Path) -> bool:
    if not files:
        return True
    result = subprocess.run(
        [tool_python(), "-m", "ruff", "format", "--check", "--config", str(cwd / "backend" / "pyproject.toml"), *files],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"ruff format failed: {result.stderr.strip()}")
    return result.returncode == 0


def formatting_is_new(current_ok: bool, baseline_ok: bool, baseline_exists: bool) -> bool:
    return not current_ok and (not baseline_exists or baseline_ok)


def run_ruff(files: list[str]) -> int:
    if not files:
        print("No backend Python files to check; skipping Ruff.")
        return 0
    relative = [Path(file).resolve().relative_to(ROOT.resolve()).as_posix() for file in files]
    try:
        with main_worktree() as baseline_root:
            current = diagnostic_counts("ruff", [str(ROOT / file) for file in relative], ROOT)
            baseline_files = [str(baseline_root / file) for file in relative if (baseline_root / file).is_file()]
            baseline = diagnostic_counts("ruff", baseline_files, baseline_root) if baseline_files else Counter()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    new = compare_diagnostics(current, baseline)
    if new:
        print("New Ruff diagnostics detected:")
        for key, count in sorted(new.items()):
            print(f"  {key}: {count}")
        return 1
    try:
        with main_worktree() as baseline_root:
            for relative_path in relative:
                current_path = ROOT / relative_path
                current_ok = format_check([str(current_path)], ROOT)
                baseline_path = baseline_root / relative_path
                if not baseline_path.is_file():
                    if formatting_is_new(current_ok, True, False):
                        print(f"New Ruff formatting violation: {relative_path}")
                        return 1
                else:
                    baseline_ok = format_check([str(baseline_path)], baseline_root)
                    if formatting_is_new(current_ok, baseline_ok, True):
                        print(f"New Ruff formatting violation: {relative_path}")
                        return 1
                    if current_ok and not baseline_ok:
                        print(f"Ruff formatting fixed: {relative_path}")
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print("Ruff diagnostics and formatting passed against the main branch baseline.")
    return 0


def run_full_ruff() -> int:
    lint = run_command([tool_python(), "-m", "ruff", "check", "--config", str(BACKEND / "pyproject.toml"), str(BACKEND)])
    if lint != 0:
        return lint
    return run_command([tool_python(), "-m", "ruff", "format", "--check", "--config", str(BACKEND / "pyproject.toml"), str(BACKEND)])


def run_full_mypy() -> int:
    return run_command([tool_python(), "-m", "mypy", "--config-file", str(BACKEND / "mypy.ini"), str(BACKEND / "server.py")])


def run_mypy() -> int:
    files = staged_python_files()
    if not files:
        print("No staged backend Python files; skipping mypy.")
        return 0
    print(f"Comparing mypy diagnostics for {len(files)} staged backend Python file(s) against main...")
    relative = [Path(file).resolve().relative_to(ROOT.resolve()).as_posix() for file in files]
    try:
        with main_worktree() as baseline_root:
            current = diagnostic_counts("mypy", [str(ROOT / file) for file in relative], ROOT)
            baseline_files = [str(baseline_root / file) for file in relative if (baseline_root / file).is_file()]
            baseline = diagnostic_counts("mypy", baseline_files, baseline_root) if baseline_files else Counter()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    new = compare_diagnostics(current, baseline)
    if new:
        print("New Mypy diagnostics detected:")
        for key, count in sorted(new.items()):
            print(f"  {key}: {count}")
        return 1
    print("Mypy passed against the main branch baseline.")
    return 0


def mypy_diagnostics(files: list[str], identity_files: list[str]) -> Counter[str]:
    result = subprocess.run(
        [
            tool_python(),
            "-m",
            "mypy",
            "--no-pretty",
            "--config-file",
            str(BACKEND / "mypy.ini"),
            *files,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    diagnostics = Counter()
    pattern = re.compile(r"^(.*?):\d+: error: (.*?)(?: \[(\w+)\])?$")
    identity_by_path = {str(Path(path).resolve()): original for path, original in zip(files, identity_files)}
    for line in result.stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            path = identity_by_path.get(str(Path(match.group(1)).resolve()), match.group(1))
            diagnostics[diagnostic_fingerprint(path, match.group(3) or "mypy", match.group(2))] += 1
    return diagnostics


def main() -> int:
    configure_console_encoding()
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/backend_quality.py [ruff|mypy|update-ruff-baseline]",
            file=sys.stderr,
        )
        return 2
    action = sys.argv[1]
    if action == "ruff":
        return run_ruff(sys.argv[2:])
    if action == "mypy":
        return run_mypy()
    if action == "full-ruff":
        return run_full_ruff()
    if action == "full-mypy":
        return run_full_mypy()
    if action == "update-ruff-baseline":
        return update_baseline()
    print(f"Unknown action: {action}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
