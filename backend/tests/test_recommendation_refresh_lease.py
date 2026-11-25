import json
from pathlib import Path

import pytest
from recommendation_refresh_lease import (
    acquire_refresh_lease,
    release_refresh_lease,
    renew_refresh_lease,
)


class Result:
    def __init__(self, row=None, rowcount=0):
        self.row = row
        self.rowcount = rowcount

    def mappings(self):
        return self

    def fetchone(self):
        return self.row


class LeaseDbFake:
    def __init__(self, *, row=None, rowcount=0):
        self.row = row
        self.rowcount = rowcount
        self.calls = []

    async def execute(self, statement, params):
        self.calls.append((str(statement), params))
        return Result(self.row, self.rowcount)


@pytest.mark.asyncio
async def test_acquire_returns_generation_and_owner():
    db = LeaseDbFake(
        row={
            "generation": 4,
            "owner_token": "owner-2",
            "lease_expires_at": "expiry",
        }
    )

    lease = await acquire_refresh_lease(db, "candidate-1", "owner-2", 60)

    assert lease.acquired is True
    assert lease.generation == 4
    assert lease.owner_token == "owner-2"
    assert db.calls[0][1] == {
        "candidate_id": "candidate-1",
        "owner_token": "owner-2",
        "lease_duration_seconds": 60,
    }


@pytest.mark.asyncio
async def test_acquire_rejected_without_returning_row_or_generation():
    db = LeaseDbFake(row=None)

    lease = await acquire_refresh_lease(db, "candidate-1", "owner-2", 60)

    assert lease.acquired is False
    assert lease.generation is None
    assert "<= now()" in db.calls[0][0]
    assert "INTERVAL '1 second'" in db.calls[0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [0, -1])
async def test_acquire_rejects_non_positive_duration_without_sql(duration):
    db = LeaseDbFake()

    with pytest.raises(ValueError, match="greater than zero"):
        await acquire_refresh_lease(db, "candidate-1", "owner-1", duration)

    assert db.calls == []


@pytest.mark.asyncio
async def test_expired_reacquisition_returns_higher_generation():
    db = LeaseDbFake(row={"generation": 9, "owner_token": "owner-2", "lease_expires_at": None})

    lease = await acquire_refresh_lease(db, "candidate-1", "owner-2", 60)

    assert lease.acquired is True
    assert lease.generation == 9
    assert "generation + 1" in db.calls[0][0]


@pytest.mark.asyncio
async def test_renew_succeeds_for_matching_owner_and_generation():
    db = LeaseDbFake(rowcount=1)

    assert await renew_refresh_lease(db, "candidate-1", "owner-1", 3, 60)
    sql, params = db.calls[0]
    assert params == {
        "candidate_id": "candidate-1",
        "owner_token": "owner-1",
        "generation": 3,
        "lease_duration_seconds": 60,
    }
    assert "owner_token = CAST(:owner_token AS uuid)" in sql
    assert "generation = :generation" in sql
    assert "lease_expires_at > now()" in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [0, -1])
async def test_renew_rejects_non_positive_duration_without_sql(duration):
    db = LeaseDbFake()

    with pytest.raises(ValueError, match="greater than zero"):
        await renew_refresh_lease(db, "candidate-1", "owner-1", 3, duration)

    assert db.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("rowcount", [0])
async def test_renew_rejects_wrong_owner_or_stale_generation(rowcount):
    db = LeaseDbFake(rowcount=rowcount)

    assert not await renew_refresh_lease(db, "candidate-1", "wrong-owner", 2, 60)


@pytest.mark.asyncio
async def test_release_succeeds_for_matching_owner_and_generation():
    db = LeaseDbFake(rowcount=1)

    assert await release_refresh_lease(db, "candidate-1", "owner-1", 3)
    sql, params = db.calls[0]
    assert params == {
        "candidate_id": "candidate-1",
        "owner_token": "owner-1",
        "generation": 3,
    }
    assert "owner_token = CAST(:owner_token AS uuid)" in sql
    assert "generation = :generation" in sql
    assert "owner_token = NULL" in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("owner,generation", [("wrong-owner", 3), ("owner-1", 2)])
async def test_release_rejects_wrong_owner_or_stale_generation(owner, generation):
    db = LeaseDbFake(rowcount=0)

    assert not await release_refresh_lease(db, "candidate-1", owner, generation)


def test_lease_migration_schema_and_railway_registration():
    root = Path(__file__).parents[1]
    migration = (root / "migrations" / "20261120_candidate_recommendation_refresh_leases.sql").read_text(
        encoding="utf-8"
    )
    assert "candidate_id UUID PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE" in migration
    assert "generation BIGINT NOT NULL DEFAULT 0" in migration
    assert "owner_token UUID NULL" in migration
    assert "lease_expires_at TIMESTAMPTZ NULL" in migration
    assert "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()" in migration

    railway = json.loads((root / "railway.json").read_text(encoding="utf-8"))
    commands = railway["deploy"]["preDeployCommand"]
    target = (
        'psql "$DATABASE_URL" -v ON_ERROR_STOP=1 '
        "-f migrations/20261120_candidate_recommendation_refresh_leases.sql"
    )
    assert commands.count(target) == 1
    freshness = (
        'psql "$DATABASE_URL" -v ON_ERROR_STOP=1 '
        "-f migrations/20261118_candidate_recommendation_freshness.sql"
    )
    durable_indexing = (
        'psql "$DATABASE_URL" -v ON_ERROR_STOP=1 '
        "-f migrations/20261121_job_indexing_work.sql"
    )
    assert commands.count(freshness) == 1
    assert commands.count(durable_indexing) == 1
    assert commands.index(freshness) < commands.index(target) < commands.index(durable_indexing)
