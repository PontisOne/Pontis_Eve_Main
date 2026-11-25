import asyncio
import time

import pytest
from app.job_ingestion import durable_indexing


class _WorkerResult:
    rowcount = 1

    def mappings(self):
        return self

    def first(self):
        return {
            "operation": "delete",
            "desired_revision": "delete:job-1",
            "claim_token": "token-1",
            "attempt_count": 0,
        }


class _WorkerDB:
    def __init__(self, *, fail_persistence=False):
        self.fail_persistence = fail_persistence
        self.statements = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def execute(self, statement, *_args, **_kwargs):
        self.statements.append(str(statement))
        if "SET status=CASE" in str(statement) and self.fail_persistence:
            raise RuntimeError("failure-state persistence failed")
        return _WorkerResult()

    async def commit(self):
        return None


class _WorkerEngine:
    def connect(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


async def _run_worker_failure(monkeypatch, *, fail_persistence=False, fail_release=False):
    release_calls = []
    async def claim(_db):
        return {"target_job_id": "job-1", "claim_token": "token-1", "desired_revision": "delete:job-1", "attempt_count": 0}

    async def acquire(*_args):
        return True

    async def current(*_args):
        return True

    async def processing(*_args):
        raise RuntimeError("processing failed")

    monkeypatch.setattr(
        durable_indexing, "_claim", claim
    )
    monkeypatch.setattr(durable_indexing, "_acquire_lock", acquire)
    monkeypatch.setattr(durable_indexing, "_claim_is_current", current)
    monkeypatch.setattr(durable_indexing, "_run_sync_safely", processing)

    async def release(*_args):
        release_calls.append(True)
        if fail_release:
            raise RuntimeError("lock release failed")

    monkeypatch.setattr(durable_indexing, "_release_lock", release)
    monkeypatch.setattr(
        "server.engine", _WorkerEngine(), raising=False
    )
    db = _WorkerDB(fail_persistence=fail_persistence)
    return await durable_indexing.process_one_work(lambda: db), release_calls


@pytest.mark.asyncio
async def test_processing_failure_persists_retry_state_and_releases_lock(monkeypatch):
    result, release_calls = await _run_worker_failure(monkeypatch)
    assert result is True
    assert release_calls == [True]


@pytest.mark.asyncio
async def test_failure_persistence_error_preserves_processing_error(monkeypatch):
    with pytest.raises(RuntimeError, match="processing failed") as raised:
        await _run_worker_failure(monkeypatch, fail_persistence=True)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "failure-state persistence failed" in str(raised.value.__cause__)


@pytest.mark.asyncio
async def test_lock_release_error_does_not_replace_processing_failure(monkeypatch):
    result, release_calls = await _run_worker_failure(monkeypatch, fail_release=True)
    assert result is True
    assert release_calls == [True]


@pytest.mark.asyncio
async def test_multiple_cleanup_failures_keep_processing_error_primary(monkeypatch):
    with pytest.raises(RuntimeError, match="processing failed") as raised:
        await _run_worker_failure(
            monkeypatch, fail_persistence=True, fail_release=True
        )
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "failure-state persistence failed" in str(raised.value.__cause__)


@pytest.mark.asyncio
async def test_ownership_loss_never_finalizes_stale_claim(monkeypatch):
    release_calls = []
    db = _WorkerDB()

    async def claim(_db):
        return {
            "target_job_id": "job-1",
            "claim_token": "old-token",
            "desired_revision": "delete:job-1",
            "attempt_count": 0,
        }

    async def heartbeat(_session_local, _job_id, _token, stop, lost):
        lost.set()
        stop.set()

    async def processing(*_args):
        await asyncio.sleep(0)

    async def release(*_args):
        release_calls.append(True)

    monkeypatch.setattr(
        _WorkerResult,
        "first",
        lambda _self: {
            "operation": "delete",
            "desired_revision": "delete:job-1",
            "claim_token": "old-token",
            "attempt_count": 0,
        },
    )
    monkeypatch.setattr(durable_indexing, "_claim", claim)
    monkeypatch.setattr(durable_indexing, "_acquire_lock", lambda *_args: _true())
    monkeypatch.setattr(durable_indexing, "_claim_is_current", lambda *_args: _true())
    monkeypatch.setattr(durable_indexing, "_heartbeat", heartbeat)
    monkeypatch.setattr(durable_indexing, "_run_sync_safely", processing)
    monkeypatch.setattr(durable_indexing, "_release_lock", release)
    monkeypatch.setattr("server.engine", _WorkerEngine(), raising=False)

    result = await durable_indexing.process_one_work(lambda: db)

    assert result is True
    assert release_calls == [True]
    assert not any("SET status='succeeded'" in statement for statement in db.statements)
    assert any("SET status=CASE" in statement for statement in db.statements)


async def _true():
    return True


@pytest.mark.asyncio
async def test_heartbeat_cleanup_swallows_only_heartbeat_cleanup_error(caplog):
    class FailedHeartbeat:
        def cancel(self):
            return None

        def __await__(self):
            async def fail():
                raise RuntimeError("heartbeat task failed")

            return fail().__await__()

    with caplog.at_level("ERROR"):
        await durable_indexing._stop_heartbeat(FailedHeartbeat(), asyncio.Event())
    assert "heartbeat task cleanup failed" in caplog.text


def test_qdrant_executor_shutdown_delegates_to_executor(monkeypatch):
    class Executor:
        def __init__(self):
            self.calls = []

        def shutdown(self, **kwargs):
            self.calls.append(kwargs)

    executor = Executor()
    monkeypatch.setattr(durable_indexing, "_QDRANT_EXECUTOR", executor)
    durable_indexing.shutdown_qdrant_executor()
    assert executor.calls == [{"wait": True, "cancel_futures": False}]


@pytest.mark.asyncio
async def test_application_shutdown_drains_executor_when_scheduler_shutdown_fails(
    monkeypatch, caplog
):
    import server
    from app.job_ingestion import scheduler

    executor_called = []
    probe_ran_before_executor_finished = []

    def fail_scheduler_shutdown():
        raise RuntimeError("scheduler shutdown failed")

    def blocking_executor_shutdown():
        time.sleep(0.05)
        executor_called.append(True)

    async def event_loop_probe():
        await asyncio.sleep(0)
        probe_ran_before_executor_finished.append(not executor_called)

    monkeypatch.setattr(scheduler, "stop_scheduler", fail_scheduler_shutdown)
    monkeypatch.setattr(
        durable_indexing,
        "shutdown_qdrant_executor",
        blocking_executor_shutdown,
    )

    with caplog.at_level("ERROR"):
        await asyncio.gather(server.on_shutdown(), event_loop_probe())

    assert executor_called == [True]
    assert probe_ran_before_executor_finished == [True]
    assert "scheduler shutdown failed" in caplog.text


@pytest.mark.asyncio
async def test_matching_task_is_registered_and_removed(monkeypatch):
    import server

    monkeypatch.setattr(server, "_MATCHING_TASKS", set())
    monkeypatch.setattr(server, "_MATCHING_SHUTTING_DOWN", False)

    async def matching(_candidate_id):
        return None

    monkeypatch.setattr(server, "_trigger_matching", matching)
    task = server._schedule_matching("candidate-1")
    assert task in server._MATCHING_TASKS
    await task
    await asyncio.sleep(0)
    assert not server._MATCHING_TASKS


@pytest.mark.asyncio
async def test_matching_task_done_callback_observes_unexpected_exception(
    monkeypatch, caplog
):
    import server

    monkeypatch.setattr(server, "_MATCHING_TASKS", set())
    monkeypatch.setattr(server, "_MATCHING_SHUTTING_DOWN", False)

    async def matching(_candidate_id):
        raise RuntimeError("unexpected matching task failure")

    monkeypatch.setattr(server, "_trigger_matching", matching)
    with caplog.at_level("ERROR"):
        server._schedule_matching("candidate-1")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert "unexpected background task failure" in caplog.text
    assert not server._MATCHING_TASKS


@pytest.mark.asyncio
async def test_matching_task_cancellation_is_logged_and_removed(monkeypatch, caplog):
    import server

    monkeypatch.setattr(server, "_MATCHING_TASKS", set())
    monkeypatch.setattr(server, "_MATCHING_SHUTTING_DOWN", False)
    started = asyncio.Event()

    async def matching(_candidate_id):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(server, "_trigger_matching", matching)
    with caplog.at_level("WARNING"):
        task = server._schedule_matching("candidate-1")
        await started.wait()
        assert task is not None
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    assert "background matching task cancelled" in caplog.text
    assert not server._MATCHING_TASKS


@pytest.mark.asyncio
async def test_matching_shutdown_drains_and_rejects_new_tasks(monkeypatch):
    import server

    monkeypatch.setattr(server, "_MATCHING_TASKS", set())
    monkeypatch.setattr(server, "_MATCHING_SHUTTING_DOWN", False)
    finished = asyncio.Event()

    async def matching(_candidate_id):
        await asyncio.sleep(0)
        finished.set()

    monkeypatch.setattr(server, "_trigger_matching", matching)
    task = server._schedule_matching("candidate-1")
    assert task is not None
    await server._shutdown_matching_tasks()
    assert finished.is_set()
    assert task.done()
    assert server._schedule_matching("candidate-2") is None


@pytest.mark.asyncio
async def test_matching_shutdown_cancels_tasks_after_timeout(monkeypatch, caplog):
    import server

    monkeypatch.setattr(server, "_MATCHING_TASKS", set())
    monkeypatch.setattr(server, "_MATCHING_SHUTTING_DOWN", False)
    monkeypatch.setattr(server, "MATCHING_SHUTDOWN_TIMEOUT_SECONDS", 0.001)
    started = asyncio.Event()

    async def matching(_candidate_id):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(server, "_trigger_matching", matching)
    task = server._schedule_matching("candidate-1")
    assert task is not None
    await started.wait()
    with caplog.at_level("WARNING"):
        await server._shutdown_matching_tasks()
    assert task.cancelled()
    assert not server._MATCHING_TASKS
    assert "shutdown timeout" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_runs_scheduler_and_qdrant_cleanup_after_matching_failure(
    monkeypatch, caplog
):
    import server
    from app.job_ingestion import durable_indexing, scheduler

    async def matching_cleanup():
        raise RuntimeError("matching cleanup failed")

    scheduler_calls = []
    executor_calls = []
    monkeypatch.setattr(server, "_shutdown_matching_tasks", matching_cleanup)
    monkeypatch.setattr(scheduler, "stop_scheduler", lambda: scheduler_calls.append(True))
    monkeypatch.setattr(
        durable_indexing,
        "shutdown_qdrant_executor",
        lambda: executor_calls.append(True),
    )
    with caplog.at_level("ERROR"):
        await server.on_shutdown()
    assert scheduler_calls == [True]
    assert executor_calls == [True]
    assert "matching task cleanup failed" in caplog.text


def test_worker_enabled_requires_all_rollout_flags(monkeypatch):
    monkeypatch.setenv("JOB_INDEXING_DURABLE_ENABLED", "true")
    monkeypatch.setenv("JOB_INDEXING_WORKER_ENABLED", "true")
    monkeypatch.setenv("JOB_INDEXING_LEGACY_WRITERS_DISABLED", "false")
    assert durable_indexing.worker_enabled() is False

    monkeypatch.setenv("JOB_INDEXING_LEGACY_WRITERS_DISABLED", "true")
    assert durable_indexing.worker_enabled() is True


@pytest.mark.parametrize(
    ("durable", "worker", "legacy", "expected_worker"),
    [
        (durable, worker, legacy, durable and worker and legacy)
        for durable in (False, True)
        for worker in (False, True)
        for legacy in (False, True)
    ],
)
def test_all_rollout_flag_combinations(
    monkeypatch, durable, worker, legacy, expected_worker
):
    monkeypatch.setenv("JOB_INDEXING_DURABLE_ENABLED", str(durable).lower())
    monkeypatch.setenv("JOB_INDEXING_WORKER_ENABLED", str(worker).lower())
    monkeypatch.setenv("JOB_INDEXING_LEGACY_WRITERS_DISABLED", str(legacy).lower())
    assert durable_indexing.durable_indexing_enabled() is durable
    assert durable_indexing.worker_enabled() is expected_worker


def test_durable_configuration_requires_worker_and_legacy_cutover(monkeypatch):
    monkeypatch.setenv("JOB_INDEXING_DURABLE_ENABLED", "true")
    monkeypatch.setenv("JOB_INDEXING_WORKER_ENABLED", "false")
    monkeypatch.setenv("JOB_INDEXING_LEGACY_WRITERS_DISABLED", "false")
    with pytest.raises(RuntimeError, match="guaranteed worker"):
        durable_indexing.validate_rollout_configuration(scheduler_enabled=True)


def test_durable_configuration_requires_scheduler(monkeypatch):
    for key in (
        "JOB_INDEXING_DURABLE_ENABLED",
        "JOB_INDEXING_WORKER_ENABLED",
        "JOB_INDEXING_LEGACY_WRITERS_DISABLED",
    ):
        monkeypatch.setenv(key, "true")
    with pytest.raises(RuntimeError, match="JOB_SCHEDULER_ENABLED=true"):
        durable_indexing.validate_rollout_configuration(scheduler_enabled=False)


def test_disabled_durable_mode_preserves_configuration_compatibility(monkeypatch):
    for key in (
        "JOB_INDEXING_DURABLE_ENABLED",
        "JOB_INDEXING_WORKER_ENABLED",
        "JOB_INDEXING_LEGACY_WRITERS_DISABLED",
    ):
        monkeypatch.setenv(key, "false")
    durable_indexing.validate_rollout_configuration(scheduler_enabled=False)


def test_retry_delay_is_exponential_and_capped(monkeypatch):
    monkeypatch.setattr(durable_indexing.random, "uniform", lambda _low, _high: 1.0)
    assert durable_indexing._retry_delay(1) == 30
    assert durable_indexing._retry_delay(2) == 60
    assert durable_indexing._retry_delay(8) == 3600


@pytest.mark.asyncio
async def test_durable_mode_never_calls_direct_qdrant_writer(monkeypatch):
    monkeypatch.setenv("JOB_INDEXING_DURABLE_ENABLED", "true")

    from app.job_ingestion import job_ingestion_service

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("durable mode must not call direct Qdrant writers")

    monkeypatch.setattr(
        "app.job_ingestion.qdrant_service.delete_job_embedding", fail_if_called
    )
    monkeypatch.setattr(
        "app.job_ingestion.qdrant_service.upsert_job_embedding", fail_if_called
    )
    await job_ingestion_service._sync_final_job_qdrant(
        None, "00000000-0000-0000-0000-000000000001"
    )


@pytest.mark.asyncio
async def test_durable_backfill_enqueues_without_any_qdrant_work(monkeypatch, capsys):
    monkeypatch.setenv("JOB_INDEXING_DURABLE_ENABLED", "true")
    from app.job_ingestion import backfill_embeddings, qdrant_service

    class Result:
        def __init__(self, *, count=0, rows=None):
            self.count = count
            self.rows = rows or []

        def scalar(self):
            return self.count

        def mappings(self):
            return self

        def fetchall(self):
            return self.rows

    class Session:
        execute_count = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, *_args, **_kwargs):
            type(self).execute_count += 1
            if type(self).execute_count == 1:
                return Result(count=2)
            return Result(
                rows=[
                    {"id": "job-1", "title": "Engineer"},
                    {"id": "job-2", "title": "Designer"},
                ]
            )

        async def commit(self):
            return None

    sessions = []

    def session_factory():
        session = Session()
        sessions.append(session)
        return session

    enqueued = []

    async def fake_enqueue(db, job_id, *, operation="upsert", **_kwargs):
        enqueued.append((job_id, operation))
        return "revision"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("durable backfill must not call Qdrant or embedding code")

    monkeypatch.setattr(backfill_embeddings, "SessionLocal", session_factory)
    monkeypatch.setattr(backfill_embeddings, "enqueue_indexing_work", fake_enqueue)
    monkeypatch.setattr(backfill_embeddings, "ensure_collection", fail_if_called)
    monkeypatch.setattr(backfill_embeddings, "generate_job_embedding", fail_if_called)
    monkeypatch.setattr(backfill_embeddings, "upsert_job_embedding", fail_if_called)
    monkeypatch.setattr(qdrant_service, "delete_job_embedding", fail_if_called)

    await backfill_embeddings.backfill()

    assert enqueued == [("job-1", "upsert"), ("job-2", "upsert")]
    assert "Failures: 0" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_authoritative_claim_check_rejects_expired_claim(monkeypatch):
    class Result:
        def first(self):
            return None

    class DB:
        async def execute(self, *_args, **_kwargs):
            return Result()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    assert await durable_indexing._claim_is_current(
        lambda: DB(), "00000000-0000-0000-0000-000000000001", "token"
    ) is False


def test_rollout_defaults_remain_disabled(monkeypatch):
    for key in (
        "JOB_INDEXING_DURABLE_ENABLED",
        "JOB_INDEXING_WORKER_ENABLED",
        "JOB_INDEXING_LEGACY_WRITERS_DISABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    assert durable_indexing.durable_indexing_enabled() is False
    assert durable_indexing.worker_enabled() is False
