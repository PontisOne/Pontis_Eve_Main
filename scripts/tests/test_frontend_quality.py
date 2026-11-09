import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import scripts.frontend_quality as quality


class CoverageLockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.lock = root / "lock"
        self.coverage = root / "coverage-tmp"
        self.original_lock = quality.COVERAGE_LOCK
        self.original_tmp = quality.COVERAGE_TMP
        quality.COVERAGE_LOCK = self.lock
        quality.COVERAGE_TMP = self.coverage
        self.env = os.environ.copy()
        self.env.update(
            {
                "PONTIS_COVERAGE_LOCK": str(self.lock),
                "PONTIS_COVERAGE_TMP": str(self.coverage),
                "PONTIS_COVERAGE_LOCK_TIMEOUT": "1",
            }
        )
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
        quality.COVERAGE_LOCK = self.original_lock
        quality.COVERAGE_TMP = self.original_tmp
        self.directory.cleanup()

    def process(self, code):
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=Path(__file__).parents[2],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.processes.append(process)
        return process

    def test_normal_acquisition_and_release(self):
        code = "import scripts.frontend_quality as q;\nwith q.coverage_run_lock(): print('ready', flush=True)"
        process = self.process(code)
        self.assertEqual(process.stdout.readline().strip(), "ready")
        self.assertEqual(process.wait(timeout=5), 0)
        self.assertFalse(self.lock.exists())

    def test_recovery_mutex_is_released_after_owner_termination(self):
        code = "import time, scripts.frontend_quality as q;\nwith q.recovery_mutex(5): print('mutex', flush=True); time.sleep(10)"
        first = self.process(code)
        self.assertEqual(first.stdout.readline().strip(), "mutex")
        first.terminate()
        first.wait(timeout=5)
        second = self.process(
            "import scripts.frontend_quality as q;\nwith q.recovery_mutex(2): print('recovered', flush=True)"
        )
        self.assertEqual(second.stdout.readline().strip(), "recovered")
        self.assertEqual(second.wait(timeout=5), 0)

    def test_recovery_mutex_timeout(self):
        first = self.process(
            "import time, scripts.frontend_quality as q;\nwith q.recovery_mutex(5): print('mutex', flush=True); time.sleep(2)"
        )
        self.assertEqual(first.stdout.readline().strip(), "mutex")
        second = self.process(
            "import scripts.frontend_quality as q;\ntry:\n with q.recovery_mutex(.2): pass\nexcept TimeoutError: print('timeout', flush=True)"
        )
        self.assertEqual(second.stdout.readline().strip(), "timeout")
        self.assertEqual(second.wait(timeout=5), 0)
        self.assertEqual(first.wait(timeout=5), 0)

    def test_concurrent_process_waits_then_succeeds(self):
        first = self.process(
            "import time, scripts.frontend_quality as q;\nwith q.coverage_run_lock(): print('ready', flush=True); time.sleep(.5)"
        )
        self.assertEqual(first.stdout.readline().strip(), "ready")
        second = self.process(
            "import scripts.frontend_quality as q;\nwith q.coverage_run_lock(): print('second', flush=True)"
        )
        self.assertEqual(second.stdout.readline().strip(), "second")
        self.assertEqual(first.wait(timeout=5), 0)
        self.assertEqual(second.wait(timeout=5), 0)

    def test_active_old_lock_is_not_removed(self):
        first = self.process(
            "import time, scripts.frontend_quality as q;\nwith q.coverage_run_lock(): print('ready', flush=True); time.sleep(2)"
        )
        self.assertEqual(first.stdout.readline().strip(), "ready")
        old = time.time() - 7200
        os.utime(self.lock, (old, old))
        second = self.process(
            "import scripts.frontend_quality as q;\ntry:\n with q.coverage_run_lock(): pass\nexcept RuntimeError as e: print('timeout', flush=True)"
        )
        self.assertEqual(second.stdout.readline().strip(), "timeout")
        self.assertTrue(self.lock.exists())
        self.assertEqual(first.wait(timeout=5), 0)
        self.assertEqual(second.wait(timeout=5), 0)

    def test_dead_process_lock_is_recovered(self):
        self.lock.mkdir()
        (self.lock / "owner").write_text(json.dumps({"pid": 99999999, "token": "dead"}))
        process = self.process(
            "import scripts.frontend_quality as q;\nwith q.coverage_run_lock(): print('recovered', flush=True)"
        )
        self.assertEqual(process.stdout.readline().strip(), "recovered")
        self.assertEqual(process.wait(timeout=5), 0)
        self.assertFalse(self.lock.exists())

    def test_two_waiters_recover_dead_lock_without_deleting_new_owner(self):
        self.lock.mkdir()
        (self.lock / "owner").write_text(json.dumps({"pid": 99999999, "token": "dead"}))
        code = "import time, scripts.frontend_quality as q;\nwith q.coverage_run_lock(): print('acquired', flush=True); time.sleep(.2)"
        first = self.process(code)
        second = self.process(code)
        self.assertEqual(first.stdout.readline().strip(), "acquired")
        self.assertEqual(second.stdout.readline().strip(), "acquired")
        self.assertEqual(first.wait(timeout=5), 0)
        self.assertEqual(second.wait(timeout=5), 0)
        self.assertFalse(self.lock.exists())

    def test_recovery_does_not_quarantine_replaced_owner(self):
        self.lock.mkdir()
        stale = {"pid": 99999999, "token": "stale"}
        (self.lock / "owner").write_text(json.dumps(stale))
        observed = json.loads((self.lock / "owner").read_text())
        shutil = __import__("shutil")
        shutil.rmtree(self.lock)
        self.lock.mkdir()
        current = {"pid": os.getpid(), "token": "new-owner"}
        (self.lock / "owner").write_text(json.dumps(current))
        quality.recover_dead_lock(observed, 0.125)
        self.assertTrue(self.lock.exists())
        self.assertEqual(json.loads((self.lock / "owner").read_text()), current)
        shutil.rmtree(self.lock)

    def test_stale_recovery_uses_the_remaining_timeout(self):
        observed = {"pid": 99999999, "token": "stale"}
        captured = []

        @quality.contextmanager
        def fake_mutex(timeout):
            captured.append(timeout)
            yield

        with mock.patch.object(quality, "recovery_mutex", fake_mutex):
            quality.recover_dead_lock(observed, 0.125)
        self.assertEqual(captured, [0.125])

    def test_owner_metadata_failure_does_not_leave_lock(self):
        with mock.patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "ownership metadata"):
                with quality.coverage_run_lock():
                    pass
        self.assertFalse(self.lock.exists())

    def test_failed_execution_releases_lock_and_cleans_after_acquisition(self):
        self.coverage.mkdir(parents=True)
        (self.coverage / "artifact").write_text("old")
        process = self.process(
            "import scripts.frontend_quality as q;\ntry:\n with q.coverage_run_lock(): raise RuntimeError('vitest failed')\nexcept RuntimeError: print('failed', flush=True)"
        )
        self.assertEqual(process.stdout.readline().strip(), "failed")
        self.assertEqual(process.wait(timeout=5), 0)
        self.assertFalse(self.lock.exists())
        self.assertFalse((self.coverage / "artifact").exists())

    def test_non_owner_cannot_release_lock(self):
        self.lock.mkdir()
        token = "owner-token"
        (self.lock / "owner").write_text(json.dumps({"pid": os.getpid(), "token": token}))
        quality.COVERAGE_LOCK = self.lock
        quality.release_coverage_lock("different-token")
        self.assertTrue(self.lock.exists())
        quality.release_coverage_lock(token)
        self.assertFalse(self.lock.exists())

    def test_lock_acquisition_timeout_is_actionable(self):
        self.lock.mkdir()
        (self.lock / "owner").write_text(json.dumps({"pid": os.getpid(), "token": "active"}))
        process = self.process(
            "import scripts.frontend_quality as q;\ntry:\n with q.coverage_run_lock(): pass\nexcept RuntimeError as e: print(str(e), flush=True)"
        )
        message = process.stdout.readline()
        self.assertIn("Timed out", message)
        self.assertIn(str(self.lock), message)
        self.assertEqual(process.wait(timeout=5), 0)
        shutil = __import__("shutil")
        shutil.rmtree(self.lock)


if __name__ == "__main__":
    unittest.main()
