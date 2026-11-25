"""Backfill Qdrant embeddings for active/open/published jobs.

Safe to run repeatedly: each job maps to one deterministic Qdrant point ID.
Use ``--reindex`` only when deliberately replacing an incompatible embedding version.

Usage (from backend/):
    python -m app.job_ingestion.backfill_embeddings --reindex
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy import text

from app.job_ingestion.embedding_service import generate_job_embedding
from app.job_ingestion.job_ingestion_service import (
    _EMBEDDING_JOB_COLUMNS,
    is_job_indexable,
)
from app.job_ingestion.qdrant_service import (
    ensure_collection,
    upsert_job_embedding,
)
from app.job_ingestion.durable_indexing import (
    durable_indexing_enabled,
    enqueue_indexing_work,
)
from server import SessionLocal

BATCH_SIZE = 100
logger = logging.getLogger(__name__)

# This intentionally mirrors candidate_job_matching_service's DB validation.
ACTIVE_JOB_WHERE = """
    is_active IS TRUE
    AND LOWER(COALESCE(status, '')) IN ('active', 'open', 'published')
    AND LOWER(COALESCE(job_status, '')) IN ('active', 'open', 'published')
"""


async def backfill(*, reindex: bool = False):
    if not durable_indexing_enabled():
        ensure_collection()
    async with SessionLocal() as db:
        count_row = await db.execute(
            text(f"SELECT COUNT(*) FROM job_descriptions WHERE {ACTIVE_JOB_WHERE}"),
        )
        total = count_row.scalar() or 0

    logger.info("[backfill] active jobs scanned=%d", total)
    print(f"Total active jobs to scan: {total}")
    succeeded = failed = offset = 0
    while offset < total:
        async with SessionLocal() as db:
            rows = await db.execute(
                text(f"""
                    SELECT {", ".join(_EMBEDDING_JOB_COLUMNS)}
                    FROM job_descriptions
                    WHERE {ACTIVE_JOB_WHERE}
                    ORDER BY id
                    LIMIT :limit OFFSET :offset
                """),
                {"limit": BATCH_SIZE, "offset": offset},
            )
            jobs = rows.mappings().fetchall()

        logger.info("[backfill] jobs selected for embedding=%d", len(jobs))
        for job in jobs:
            job_dict = dict(job)
            job_id = str(job_dict["id"])
            try:
                if durable_indexing_enabled():
                    async with SessionLocal() as work_db:
                        await enqueue_indexing_work(work_db, job_id, operation="upsert")
                        await work_db.commit()
                    succeeded += 1
                    continue
                if not is_job_indexable(job_dict):
                    from app.job_ingestion.qdrant_service import delete_job_embedding

                    delete_job_embedding(job_id)
                    continue
                embedding = generate_job_embedding(job_dict)
                upsert_job_embedding(job_id, embedding, job_dict, reindex=reindex)
                succeeded += 1
            except Exception as exc:
                logger.exception("[backfill] failed job_id=%s", job_id)
                print(f"  FAILED job_id={job_id}: {exc}")
                failed += 1

        offset += BATCH_SIZE
        logger.info(
            "[backfill] Qdrant points written=%d failures=%d", succeeded, failed
        )
        print(
            f"  Processed {min(offset, total)}/{total} - succeeded={succeeded} failed={failed}"
        )

    print(f"\nDone. Successfully indexed: {succeeded} | Failures: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="replace incompatible vector versions explicitly",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(backfill(reindex=args.reindex))
