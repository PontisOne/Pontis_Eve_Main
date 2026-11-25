"""Durable PostgreSQL-to-Qdrant indexing work queue and worker."""

import asyncio
import hashlib
import logging
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from app.job_ingestion.job_ingestion_service import (
    is_job_indexable,
    load_persisted_job_for_embedding,
)
from sqlalchemy import text

logger = logging.getLogger(__name__)
_QDRANT_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("JOB_INDEXING_QDRANT_WORKERS", "2"))),
    thread_name_prefix="job-indexing",
)


class _OwnershipLostError(RuntimeError):
    """The worker can no longer prove that it owns the claimed work item."""


def shutdown_qdrant_executor() -> None:
    """Drain and close the executor during application shutdown."""
    _QDRANT_EXECUTOR.shutdown(wait=True, cancel_futures=False)

CLAIM_LEASE_SECONDS = int(os.getenv("JOB_INDEXING_CLAIM_LEASE_SECONDS", "600"))
HEARTBEAT_SECONDS = int(os.getenv("JOB_INDEXING_HEARTBEAT_SECONDS", "120"))
LOCK_TIMEOUT_SECONDS = float(os.getenv("JOB_INDEXING_LOCK_TIMEOUT_SECONDS", "15"))
MAX_ATTEMPTS = int(os.getenv("JOB_INDEXING_MAX_ATTEMPTS", "8"))
BASE_RETRY_SECONDS = int(os.getenv("JOB_INDEXING_BASE_RETRY_SECONDS", "30"))
MAX_RETRY_SECONDS = int(os.getenv("JOB_INDEXING_MAX_RETRY_SECONDS", "3600"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_VERSION = os.getenv("EMBEDDING_VERSION", "v2_structured")


def durable_indexing_enabled() -> bool:
    return os.getenv("JOB_INDEXING_DURABLE_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def worker_enabled() -> bool:
    return durable_indexing_enabled() and os.getenv(
        "JOB_INDEXING_WORKER_ENABLED", "false"
    ).strip().lower() in {"1", "true", "yes", "on"} and os.getenv(
        "JOB_INDEXING_LEGACY_WRITERS_DISABLED", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}


def validate_rollout_configuration(*, scheduler_enabled: bool) -> None:
    """Reject durable indexing configurations that cannot drain their queue."""
    if not durable_indexing_enabled():
        return

    missing = []
    if not _env_flag("JOB_INDEXING_WORKER_ENABLED"):
        missing.append("JOB_INDEXING_WORKER_ENABLED=true")
    if not _env_flag("JOB_INDEXING_LEGACY_WRITERS_DISABLED"):
        missing.append("JOB_INDEXING_LEGACY_WRITERS_DISABLED=true")
    if not scheduler_enabled:
        missing.append("JOB_SCHEDULER_ENABLED=true")
    if missing:
        raise RuntimeError(
            "Unsafe durable job indexing configuration: "
            "JOB_INDEXING_DURABLE_ENABLED=true requires "
            + ", ".join(missing)
            + ". Durable indexing would enqueue work without a guaranteed worker."
        )


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def job_revision(job: dict[str, Any]) -> str:
    """Hash the exact bounded production embedding text and configuration."""
    from app.job_ingestion.embedding_service import build_job_text

    canonical_text = build_job_text(job)
    value = f"{EMBEDDING_MODEL_NAME}\0{EMBEDDING_VERSION}\0{canonical_text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def enqueue_indexing_work(
    db, job_id: str, *, operation: str = "upsert", revision: str | None = None,
    lock_held: bool = False,
) -> str:
    """Coalesce work inside the caller's existing PostgreSQL transaction."""
    job_id = str(job_id)
    if operation == "upsert":
        job = await load_persisted_job_for_embedding(db, job_id)
        revision = revision or job_revision(job)
        params = {"job_id": job_id, "revision": revision}
        sql = "".join(
            (
                "\n",
                "            INSERT INTO job_indexing_work\n",
                "                (target_job_id, job_id, operation, desired_revision, "
                "status,\n",
                "                 next_retry_at, created_at, updated_at)\n",
                "            VALUES (CAST(:job_id AS UUID), CAST(:job_id AS UUID), "
                "'upsert', :revision,\n",
                "                    'pending', NOW(), NOW(), NOW())\n",
                "            ON CONFLICT (target_job_id) DO UPDATE SET\n",
                "                job_id = EXCLUDED.job_id,\n",
                "                operation = 'upsert',\n",
                "                desired_revision = EXCLUDED.desired_revision,\n",
                "                status = CASE\n",
                "                    WHEN job_indexing_work.status = 'processing'\n",
                "                         AND job_indexing_work.desired_revision = "
                "EXCLUDED.desired_revision\n",
                "                    THEN job_indexing_work.status ELSE 'pending' "
                "END,\n",
                "                next_retry_at = CASE\n",
                "                    WHEN job_indexing_work.status = 'processing'\n",
                "                         AND job_indexing_work.desired_revision = "
                "EXCLUDED.desired_revision\n",
                "                    THEN job_indexing_work.next_retry_at ELSE NOW() "
                "END,\n",
                "                last_error = NULL, last_error_at = NULL,\n",
                "                updated_at = NOW()\n",
                "        ",
            )
        )
    elif operation == "delete":
        revision = revision or f"delete:{job_id}"
        params = {"job_id": job_id, "revision": revision}
        sql = """
            INSERT INTO job_indexing_work
                (target_job_id, job_id, operation, desired_revision, status,
                 next_retry_at, created_at, updated_at)
            VALUES (CAST(:job_id AS UUID), NULL, 'delete', :revision,
                    'pending', NOW(), NOW(), NOW())
            ON CONFLICT (target_job_id) DO UPDATE SET
                job_id = NULL,
                operation = 'delete',
                desired_revision = EXCLUDED.desired_revision,
                status = CASE WHEN job_indexing_work.status = 'processing'
                    THEN job_indexing_work.status ELSE 'pending' END,
                next_retry_at = CASE WHEN job_indexing_work.status = 'processing'
                    THEN job_indexing_work.next_retry_at ELSE NOW() END,
                last_error = NULL, last_error_at = NULL,
                updated_at = NOW()
        """
    else:
        raise ValueError(f"Unsupported indexing operation: {operation}")
    if lock_held:
        await db.execute(text(sql), params)
    else:
        async with job_advisory_lock(job_id):
            await db.execute(text(sql), params)
    return revision


def _lock_key(job_id: str) -> int:
    # PostgreSQL bigint advisory-lock key; stable across processes.
    return int.from_bytes(
        hashlib.sha256(f"eve-job-index:{job_id}".encode()).digest()[:8],
        "big",
        signed=True,
    )


async def _acquire_lock(connection, job_id: str) -> bool:
    key = _lock_key(job_id)
    deadline = asyncio.get_running_loop().time() + LOCK_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        result = await connection.execute(
            text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
        )
        if bool(result.scalar()):
            return True
        await asyncio.sleep(0.1)
    return False


async def _release_lock(connection, job_id: str) -> None:
    await connection.execute(
        text("SELECT pg_advisory_unlock(:key)"), {"key": _lock_key(job_id)}
    )


@asynccontextmanager
async def job_advisory_lock(job_id: str):
    """Hold the per-job session lock on a dedicated connection."""
    from server import engine

    async with engine.connect() as connection:
        acquired = await _acquire_lock(connection, job_id)
        if not acquired:
            raise TimeoutError(f"Timed out acquiring job indexing lock: {job_id}")
        try:
            yield connection
        finally:
            await _release_lock(connection, job_id)


@asynccontextmanager
async def session_job_advisory_lock(db, job_id: str):
    """Hold the session-level lock on the same checked-out DB session."""
    connection = await db.connection()
    acquired = await _acquire_lock(connection, str(job_id))
    if not acquired:
        raise TimeoutError(f"Timed out acquiring job indexing lock: {job_id}")
    try:
        yield
    finally:
        await _release_lock(connection, str(job_id))


def _retry_delay(attempt: int) -> int:
    base = min(MAX_RETRY_SECONDS, BASE_RETRY_SECONDS * (2 ** max(0, attempt - 1)))
    return max(1, int(base * random.uniform(0.8, 1.2)))


async def _run_sync_safely(function, *args):
    """Run blocking indexing work without releasing its lock mid-call."""
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(_QDRANT_EXECUTOR, function, *args)
    try:
        return await future
    except asyncio.CancelledError:
        await asyncio.shield(future)
        raise


async def _heartbeat(
    SessionLocal,
    job_id: str,
    token: str,
    stop: asyncio.Event,
    lost: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), HEARTBEAT_SECONDS)
        except TimeoutError:
            try:
                async with SessionLocal() as db:
                    result = await db.execute(
                    text(
                        "".join(
                            (
                                "\n                    UPDATE job_indexing_work\n",
                                "                    SET claim_expires_at = NOW() + "
                                "(:lease * INTERVAL '1 second'), updated_at = NOW()\n",
                                "                    WHERE target_job_id = CAST("
                                ":job_id AS UUID)\n",
                                "                      AND claim_token = CAST(:token "
                                "AS UUID)\n",
                                "                      AND status = '"
                                "processing'\n",
                                "                    ",
                            )
                        )
                    ),
                    {"job_id": job_id, "token": token, "lease": CLAIM_LEASE_SECONDS},
                )
                    await db.commit()
                if result.rowcount != 1:
                    lost.set()
                    stop.set()
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[job-indexing] heartbeat failed job_id=%s", job_id)
                lost.set()
                stop.set()
                return


async def _stop_heartbeat(heartbeat_task, heartbeat_stop: asyncio.Event) -> None:
    heartbeat_stop.set()
    if heartbeat_task is None:
        return
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("[job-indexing] heartbeat task cleanup failed")


async def _claim_is_current(SessionLocal, job_id: str, token: str) -> bool:
    """Authoritatively validate ownership immediately before external I/O."""
    async with SessionLocal() as db:
        result = await db.execute(text("""
            SELECT 1
            FROM job_indexing_work
            WHERE target_job_id = CAST(:job_id AS UUID)
              AND claim_token = CAST(:token AS UUID)
              AND status = 'processing'
              AND claim_expires_at > NOW()
        """), {"job_id": job_id, "token": token})
        return result.first() is not None


async def _claim(db) -> dict[str, Any] | None:
    result = await db.execute(text("""
        SELECT target_job_id, job_id, operation, desired_revision, attempt_count
        FROM job_indexing_work
        WHERE (status IN ('pending', 'failed') AND next_retry_at <= NOW())
           OR (status = 'processing' AND claim_expires_at < NOW())
        ORDER BY next_retry_at, updated_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    """))
    row = result.mappings().first()
    if row is None:
        await db.rollback()
        return None
    token = str(uuid.uuid4())
    await db.execute(text("""
        UPDATE job_indexing_work
        SET status = 'processing', claim_token = CAST(:token AS UUID),
            claimed_at = NOW(), claim_expires_at = NOW() + (:lease * INTERVAL '1 """
        "second'),\n"
        """            updated_at = NOW()
        WHERE target_job_id = CAST(:job_id AS UUID)
    """), {
        "job_id": row["target_job_id"],
        "token": token,
        "lease": CLAIM_LEASE_SECONDS,
    })
    await db.commit()
    return {**dict(row), "claim_token": token}


async def process_one_work(SessionLocal) -> bool:
    async with SessionLocal() as claim_db:
        work = await _claim(claim_db)
    if not work:
        return False
    job_id = str(work["target_job_id"])
    token = work["claim_token"]
    from server import engine

    async with engine.connect() as lock_connection:
        if not await _acquire_lock(lock_connection, job_id):
            async with SessionLocal() as db:
                await db.execute(
                    text(
                        "".join(
                            (
                                "\n                    UPDATE job_indexing_work SET "
                                "status='pending', claim_token=NULL,\n",
                                "                        claimed_at=NULL, "
                                "claim_expires_at=NULL,\n",
                                "                        next_retry_at=NOW() + "
                                "INTERVAL '5 seconds', updated_at=NOW()\n",
                                "                    WHERE target_job_id=CAST(:job_id "
                                "AS UUID) AND claim_token=CAST(:token AS UUID)\n",
                                "                ",
                            )
                        )
                    ),
                    {"job_id": job_id, "token": token},
                )
                await db.commit()
            return True
        heartbeat_stop = asyncio.Event()
        ownership_lost = asyncio.Event()
        heartbeat_task = None
        try:
            async with SessionLocal() as db:
                current = (await db.execute(text("""
                    SELECT operation, desired_revision, claim_token, attempt_count
                    FROM job_indexing_work WHERE target_job_id=CAST(:job_id AS UUID)
                """), {"job_id": job_id})).mappings().first()
                if not current or str(current["claim_token"]) != token:
                    return True
                if current["operation"] == "upsert":
                    try:
                        job = await load_persisted_job_for_embedding(db, job_id)
                    except RuntimeError:
                        await enqueue_indexing_work(
                            db, job_id, operation="delete", lock_held=True
                        )
                        await db.commit()
                        return True
                    if not is_job_indexable(job):
                        await enqueue_indexing_work(
                            db, job_id, operation="delete", lock_held=True
                        )
                        await db.commit()
                        return True
                    revision = job_revision(job)
                    if revision != current["desired_revision"]:
                        await enqueue_indexing_work(
                            db, job_id, revision=revision, lock_held=True
                        )
                        await db.commit()
                        return True
                    await db.execute(
                        text(
                            "".join(
                                (
                                    "\n                        UPDATE "
                                    "job_indexing_work "
                                    "SET attempt_count=attempt_count+1, "
                                    "updated_at=NOW()\n",
                                    "                        WHERE target_job_id=CAST("
                                    ":job_id AS UUID) AND claim_token=CAST("
                                    ":token AS UUID)\n",
                                    "                    ",
                                )
                            )
                        ),
                        {"job_id": job_id, "token": token},
                    )
                    await db.commit()
                    heartbeat_task = asyncio.create_task(
                        _heartbeat(
                            SessionLocal,
                            job_id,
                            token,
                            heartbeat_stop,
                            ownership_lost,
                        )
                    )
                    if ownership_lost.is_set():
                        return True
                    from app.job_ingestion.embedding_service import (
                        generate_job_embedding,
                    )
                    from app.job_ingestion.qdrant_service import (
                        ensure_collection,
                        upsert_job_embedding,
                    )
                    if ownership_lost.is_set() or not await _claim_is_current(
                        SessionLocal, job_id, token
                    ):
                        return True
                    await _run_sync_safely(ensure_collection)
                    if ownership_lost.is_set() or not await _claim_is_current(
                        SessionLocal, job_id, token
                    ):
                        return True
                    embedding = await _run_sync_safely(generate_job_embedding, job)
                    if ownership_lost.is_set() or not await _claim_is_current(
                        SessionLocal, job_id, token
                    ):
                        return True
                    await _run_sync_safely(
                        upsert_job_embedding,
                        job_id,
                        embedding,
                        job,
                    )
                else:
                    await db.execute(
                        text(
                            "".join(
                                (
                                    "\n                        UPDATE "
                                    "job_indexing_work "
                                    "SET attempt_count=attempt_count+1, "
                                    "updated_at=NOW()\n",
                                    "                        WHERE target_job_id=CAST("
                                    ":job_id AS UUID) AND claim_token=CAST("
                                    ":token AS UUID)\n",
                                    "                    ",
                                )
                            )
                        ),
                        {"job_id": job_id, "token": token},
                    )
                    await db.commit()
                    heartbeat_task = asyncio.create_task(
                        _heartbeat(
                            SessionLocal,
                            job_id,
                            token,
                            heartbeat_stop,
                            ownership_lost,
                        )
                    )
                    if ownership_lost.is_set():
                        return True
                    from app.job_ingestion.qdrant_service import delete_job_embedding
                    if ownership_lost.is_set() or not await _claim_is_current(
                        SessionLocal, job_id, token
                    ):
                        return True
                    await _run_sync_safely(delete_job_embedding, job_id)
            if ownership_lost.is_set():
                raise _OwnershipLostError(
                    f"Lost durable indexing claim during processing: {job_id}"
                )
            async with SessionLocal() as db:
                result = await db.execute(
                    text(
                        "".join(
                            (
                                "\n                    UPDATE job_indexing_work\n",
                                "                    SET status='succeeded', "
                                "last_succeeded_operation=operation,\n",
                                "                        last_succeeded_revision="
                                "desired_revision, last_succeeded_at=NOW(),\n",
                                "                        last_error=NULL, "
                                "last_error_at=NULL, claim_token=NULL,\n",
                                "                        claimed_at=NULL, "
                                "claim_expires_at=NULL, updated_at=NOW()\n",
                                "                    WHERE target_job_id=CAST(:job_id "
                                "AS UUID)\n",
                                "                      AND claim_token=CAST(:token "
                                "AS UUID)\n",
                                "                      AND desired_revision= :revision"
                                "\n",
                                "                ",
                            )
                        )
                    ), {
                    "job_id": job_id,
                    "token": token,
                    "revision": work["desired_revision"],
                })
                await db.commit()
                if result.rowcount != 1:
                    logger.warning("[job-indexing] finalization lost job_id=%s", job_id)
        except Exception as exc:
            logger.exception("[job-indexing] failed job_id=%s", job_id)
            try:
                async with SessionLocal() as db:
                    await db.execute(
                        text(
                            "".join(
                                (
                                    "\n                        UPDATE "
                                    "job_indexing_work\n",
                                    "                        SET status=CASE WHEN "
                                    "attempt_count >= :max_attempts THEN 'dead_letter' "
                                    "ELSE 'failed' END,\n",
                                    "                            last_error=:error, "
                                    "last_error_at=NOW(),\n",
                                    "                            next_retry_at=NOW() + "
                                    "(:delay * INTERVAL '1 second'),\n",
                                    "                            claim_token=NULL, "
                                    "claimed_at=NULL, claim_expires_at=NULL, "
                                    "updated_at=NOW()\n",
                                    "                        WHERE target_job_id=CAST("
                                    ":job_id "
                                    "AS UUID) AND claim_token=CAST(:token AS UUID)\n",
                                    "                    ",
                                )
                            )
                        ), {
                            "job_id": job_id,
                            "token": token,
                            "max_attempts": MAX_ATTEMPTS,
                            "delay": _retry_delay(
                                int(work.get("attempt_count") or 0) + 1
                            ),
                            "error": str(exc)[:1000],
                        }
                    )
                    await db.commit()
            except Exception as persist_exc:
                logger.exception(
                    "[job-indexing] failed to persist failure state job_id=%s",
                    job_id,
                )
                raise exc from persist_exc
        finally:
            await _stop_heartbeat(heartbeat_task, heartbeat_stop)
            try:
                await _release_lock(lock_connection, job_id)
            except Exception:
                logger.exception(
                    "[job-indexing] failed to release lock job_id=%s", job_id
                )
    return True


async def run_worker_once(SessionLocal=None) -> int:
    if not worker_enabled():
        return 0
    if SessionLocal is None:
        from server import SessionLocal as default_session
        SessionLocal = default_session
    processed = 0
    while await process_one_work(SessionLocal):
        processed += 1
    return processed


async def replay_dead_letter(
    job_id: str, SessionLocal=None, *, reset_attempt_count: bool = True
) -> bool:
    """Return one dead-letter operation to the normal worker queue."""
    if SessionLocal is None:
        from server import SessionLocal as default_session
        SessionLocal = default_session
    async with job_advisory_lock(str(job_id)):
        async with SessionLocal() as db:
            result = await db.execute(text("""
                UPDATE job_indexing_work
                SET status='pending',
                    attempt_count=CASE WHEN :reset_attempts THEN 0 ELSE """
                "attempt_count END,\n"
                """                    last_error=NULL, last_error_at=NULL,
                    next_retry_at=NOW(), updated_at=NOW()
                WHERE target_job_id=CAST(:job_id AS UUID)
                  AND status='dead_letter'
                  AND claim_token IS NULL
                RETURNING target_job_id
            """), {"job_id": str(job_id), "reset_attempts": reset_attempt_count})
            await db.commit()
            return result.first() is not None
