import os
import shutil
import subprocess
import sys
import tempfile
import time
import json
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
COVERAGE_TMP = Path(os.environ.get("PONTIS_COVERAGE_TMP", FRONTEND / "coverage" / ".tmp"))
COVERAGE_LOCK = Path(
    os.environ.get(
        "PONTIS_COVERAGE_LOCK",
        Path(tempfile.gettempdir()) / "pontis-eve-frontend-vitest.lock.d",
    )
)
COVERAGE_LOCK_TIMEOUT = float(os.environ.get("PONTIS_COVERAGE_LOCK_TIMEOUT", 30))

COMMANDS = {
    "typecheck": [
        FRONTEND / "node_modules" / "typescript" / "bin" / "tsc",
        "--noEmit",
        "-p",
        "tsconfig.json",
    ],
    "test": [
        FRONTEND / "node_modules" / "vitest" / "vitest.mjs",
        "run",
        "--coverage",
    ],
}


@contextmanager
def recovery_mutex(timeout: float):
    """Hold an OS-released cross-platform mutex for the critical section."""
    path = COVERAGE_LOCK.with_name(f"{COVERAGE_LOCK.name}.recovery.lock")
    deadline = time.monotonic() + timeout
    handle = path.open("a+b")
    locked = False
    try:
        while time.monotonic() < deadline:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    if handle.read(1) == b"":
                        handle.seek(0)
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except (BlockingIOError, OSError):
                time.sleep(0.05)
        if not locked:
            raise TimeoutError(f"Timed out acquiring recovery mutex {path}")
        yield
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        else:
            handle.close()


@contextmanager
def coverage_run_lock():
    """Serialize covered runs with bounded, owner-verified lock handling."""
    token = uuid.uuid4().hex
    owner = COVERAGE_LOCK / "owner"
    deadline = time.monotonic() + COVERAGE_LOCK_TIMEOUT
    acquired = False
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            mutex = recovery_mutex(remaining)
            mutex.__enter__()
        except (TimeoutError, OSError):
            time.sleep(0.1)
            continue
        try:
            COVERAGE_LOCK.mkdir()
        except FileExistsError:
            pass
        else:
            try:
                owner.write_text(
                    json.dumps({"pid": os.getpid(), "token": token}), encoding="utf-8"
                )
            except OSError as exc:
                shutil.rmtree(COVERAGE_LOCK, ignore_errors=True)
                raise RuntimeError(
                    f"Could not write coverage lock ownership metadata: {exc}"
                ) from exc
            acquired = True
        finally:
            mutex.__exit__(None, None, None)
        if acquired:
            break
        try:
            metadata = json.loads(owner.read_text(encoding="utf-8"))
            pid = int(metadata["pid"])
            process_is_alive(pid)
        except ProcessLookupError:
            recover_dead_lock(metadata, max(0, deadline - time.monotonic()))
            continue
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError, PermissionError, OSError):
            # Invalid metadata is not safe to recover; allow the bounded
            # timeout to report the lock.
            pass
        time.sleep(0.1)
    if not acquired:
        raise RuntimeError(
            f"Timed out after {COVERAGE_LOCK_TIMEOUT:g}s waiting for coverage lock "
            f"{COVERAGE_LOCK}. Verify the owner process or remove a confirmed stale lock."
        )

    try:
        if COVERAGE_TMP.exists():
            for artifact in COVERAGE_TMP.iterdir():
                if artifact.is_file() or artifact.is_symlink():
                    artifact.unlink()
                elif artifact.is_dir():
                    shutil.rmtree(artifact)
        yield
    finally:
        release_coverage_lock(token)


def release_coverage_lock(token: str) -> None:
    """Release the lock only when its PID and random owner token match."""
    try:
        metadata = json.loads((COVERAGE_LOCK / "owner").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if metadata.get("pid") == os.getpid() and metadata.get("token") == token:
        shutil.rmtree(COVERAGE_LOCK, ignore_errors=True)


def process_is_alive(pid: int) -> bool:
    """Check a PID portably; permission failures mean the process is alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise
    except PermissionError:
        return True
    except OSError as exc:
        if getattr(exc, "winerror", None) == 87:
            raise ProcessLookupError(pid) from exc
        raise
    return True


def recover_dead_lock(expected_metadata: dict, timeout: float) -> None:
    """Quarantine only the stale lock whose metadata was previously verified."""
    try:
        with recovery_mutex(timeout):
            try:
                current = json.loads((COVERAGE_LOCK / "owner").read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return
            if current != expected_metadata:
                return
            quarantine = COVERAGE_LOCK.with_name(
                f"{COVERAGE_LOCK.name}.stale-{uuid.uuid4().hex}"
            )
            try:
                COVERAGE_LOCK.rename(quarantine)
            except (FileNotFoundError, OSError):
                return
            shutil.rmtree(quarantine, ignore_errors=True)
    except (TimeoutError, OSError):
        return


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "staged-test"):
        print("Usage: frontend_quality.py {typecheck|staged-test|test}", file=sys.stderr)
        return 2

    if sys.argv[1] == "staged-test":
        staged = [Path(name).as_posix() for name in sys.argv[2:] if ".test." in name or ".spec." in name]
        if not staged:
            return 0
        command = [
            FRONTEND / "node_modules" / "vitest" / "vitest.mjs",
            "run",
            *[Path(name).relative_to("frontend").as_posix() for name in staged],
        ]
    else:
        command = COMMANDS[sys.argv[1]].copy()
    launcher = command[0]
    if not launcher.is_file():
        print(
            "Frontend dependencies are not installed. Run "
            "`corepack yarn --cwd frontend install --frozen-lockfile`.",
            file=sys.stderr,
        )
        return 1

    run = coverage_run_lock() if sys.argv[1] == "test" else nullcontext()
    with run:
        result = subprocess.run(
            ["node", *(str(part) for part in command)],
            cwd=FRONTEND,
            check=False,
        )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
