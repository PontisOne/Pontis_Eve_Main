import pytest

from app.job_ingestion import scheduler


class Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class DB:
    def __init__(self):
        self.calls = []

    async def execute(self, statement, params=None):
        sql = str(statement).lower()
        self.calls.append((sql, params or {}))
        if "select id, ats_job_id" in sql:
            return Result([("db-1", "f-1")])
        return Result([])

    async def commit(self):
        pass


class FantasticJobBoardDB:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def execute(self, statement, params=None):
        sql = str(statement).lower()
        self.calls.append((sql, params or {}))
        if "select id, ats_job_id, structured_data" in sql:
            return Result(self.rows)
        return Result([])

    async def commit(self):
        pass


@pytest.mark.asyncio
async def test_confirmed_provider_id_closes_and_hides_job():
    db = DB()
    closed = await scheduler._close_confirmed_jobs(
        db, ats_type="fantastic", ats_job_ids={"f-1"}
    )
    assert closed == ["db-1"]
    assert any("update job_descriptions" in sql for sql, _ in db.calls)
    assert any("update candidate_job_recommendations" in sql for sql, _ in db.calls)


@pytest.mark.asyncio
async def test_unrelated_provider_id_changes_nothing():
    class EmptyDB(DB):
        async def execute(self, statement, params=None):
            sql = str(statement).lower()
            self.calls.append((sql, params or {}))
            return Result([])

    db = EmptyDB()
    assert (
        await scheduler._close_confirmed_jobs(
            db, ats_type="fantastic", ats_job_ids={"other"}
        )
        == []
    )
    assert not any("update job_descriptions" in sql for sql, _ in db.calls)


@pytest.mark.asyncio
async def test_expired_job_board_closes_only_linkedin_active_jb_rows():
    db = FantasticJobBoardDB(
        [
            (
                "db-linkedin",
                "fj-1",
                {"source": "linkedin", "fantastic_feed": "active-jb"},
            ),
            (
                "db-wellfound",
                "fj-2",
                {"source": "wellfound", "fantastic_feed": "active-jb"},
            ),
            ("db-yc", "fj-3", {"source": "ycombinator", "fantastic_feed": "active-jb"}),
            (
                "db-ats",
                "fj-1",
                {"source": "greenhouse", "fantastic_feed": "active-ats"},
            ),
            ("db-unknown", "fj-4", {"fantastic_feed": "active-jb"}),
        ]
    )

    closed, skipped = await scheduler._close_confirmed_fantastic_job_board_jobs(
        db, ats_job_ids={"fj-1", "fj-2", "fj-3", "fj-4"}
    )

    assert closed == ["db-linkedin"]
    assert skipped == 4
    assert any("update job_descriptions" in sql for sql, _ in db.calls)
    assert any("update candidate_job_recommendations" in sql for sql, _ in db.calls)


@pytest.mark.asyncio
async def test_expired_job_board_fallback_source_is_supported():
    db = FantasticJobBoardDB(
        [
            (
                "db-linkedin",
                "fj-1",
                {"fantastic": {"source": "linkedin"}, "fantastic_feed": "active-jb"},
            ),
        ]
    )

    closed, skipped = await scheduler._close_confirmed_fantastic_job_board_jobs(
        db, ats_job_ids={"fj-1"}
    )

    assert closed == ["db-linkedin"]
    assert skipped == 0


@pytest.mark.asyncio
async def test_expired_job_board_empty_ids_close_nothing():
    db = FantasticJobBoardDB([])
    assert await scheduler._close_confirmed_fantastic_job_board_jobs(
        db, ats_job_ids=set()
    ) == ([], 0)
    assert db.calls == []


@pytest.mark.asyncio
async def test_expired_job_board_cleanup_deletes_qdrant_by_database_uuid(monkeypatch):
    deleted = []

    class Client:
        async def fetch_expired_job_boards(self, time_frame):
            assert time_frame == "1d"
            return ["fj-1"]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def commit(self):
            pass

    async def close_jobs(db, *, ats_job_ids):
        assert ats_job_ids == {"fj-1"}
        return ["db-uuid-1"], 0

    def delete_embedding(job_id):
        deleted.append(job_id)

    monkeypatch.setattr(
        "app.job_ingestion.connectors.fantastic.FantasticClient", Client
    )
    monkeypatch.setattr(scheduler, "_get_session_local", lambda: lambda: Session())
    monkeypatch.setattr(
        scheduler, "_close_confirmed_fantastic_job_board_jobs", close_jobs
    )
    monkeypatch.setattr(
        "app.job_ingestion.qdrant_service.delete_job_embedding", delete_embedding
    )

    stats = await scheduler.cleanup_fantastic_expired_job_boards()

    assert stats["jobs_confirmed_closed"] == 1
    assert deleted == ["db-uuid-1"]


@pytest.mark.asyncio
async def test_expired_job_board_qdrant_failure_does_not_reopen_job(monkeypatch):
    class Client:
        async def fetch_expired_job_boards(self, _time_frame):
            return ["fj-1"]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def commit(self):
            pass

    async def close_jobs(_db, *, ats_job_ids):
        assert ats_job_ids == {"fj-1"}
        return ["db-uuid-1"], 0

    def delete_embedding(_job_id):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(
        "app.job_ingestion.connectors.fantastic.FantasticClient", Client
    )
    monkeypatch.setattr(scheduler, "_get_session_local", lambda: lambda: Session())
    monkeypatch.setattr(
        scheduler, "_close_confirmed_fantastic_job_board_jobs", close_jobs
    )
    monkeypatch.setattr(
        "app.job_ingestion.qdrant_service.delete_job_embedding", delete_embedding
    )

    stats = await scheduler.cleanup_fantastic_expired_job_boards()

    assert stats["jobs_confirmed_closed"] == 1
    assert stats["qdrant_failures"] == 1
    assert stats["errors"] == 0


@pytest.mark.asyncio
async def test_expired_job_board_api_error_closes_nothing(monkeypatch):
    class Client:
        async def fetch_expired_job_boards(self, _time_frame):
            raise RuntimeError("provider unavailable")

    def session_factory():
        raise AssertionError("database must not be opened after API failure")

    monkeypatch.setattr(
        "app.job_ingestion.connectors.fantastic.FantasticClient", Client
    )
    monkeypatch.setattr(scheduler, "_get_session_local", session_factory)

    stats = await scheduler.cleanup_fantastic_expired_job_boards()

    assert stats["jobs_confirmed_closed"] == 0
    assert stats["errors"] == 1


@pytest.mark.asyncio
async def test_combined_fantastic_cleanup_reports_job_board_failure(monkeypatch):
    class Client:
        _requests_used = 1

        async def fetch_expired_ats_ids(self, _time_frame):
            return []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def commit(self):
            pass

    async def close_jobs(_db, **_kwargs):
        return []

    async def failed_job_board_cleanup(_client):
        return {
            "jobs_checked": 2,
            "jobs_confirmed_closed": 0,
            "jobs_unresolved": 0,
            "jobs_skipped": 0,
            "qdrant_failures": 0,
            "api_requests_made": 2,
            "errors": 1,
        }

    monkeypatch.setattr(
        "app.job_ingestion.connectors.fantastic.FantasticClient", Client
    )
    monkeypatch.setattr(scheduler, "_get_session_local", lambda: lambda: Session())
    monkeypatch.setattr(scheduler, "_close_confirmed_jobs", close_jobs)
    monkeypatch.setattr(
        scheduler, "cleanup_fantastic_expired_job_boards", failed_job_board_cleanup
    )

    stats = await scheduler.cleanup_fantastic_expired_jobs()

    assert stats["errors"] == 1
    assert stats["job_board_errors"] == 1
    assert stats["job_board_jobs_checked"] == 2
    assert stats["api_requests_made"] == 2


@pytest.mark.asyncio
async def test_theirstack_client_batches_exact_ids(monkeypatch):
    calls = []

    class Client:
        async def fetch_jobs_by_ids(self, ids):
            calls.append(ids)
            return [{"id": ids[0], "closed_at": "2026-10-06T00:00:00Z"}]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def commit(self):
            pass

        async def execute(self, *_args, **_kwargs):
            return Result([(str(i),) for i in range(205)])

    monkeypatch.setattr(scheduler, "_get_session_local", lambda: lambda: Session())
    monkeypatch.setattr(
        "app.job_ingestion.connectors.theirstack.TheirStackClient", Client
    )
    monkeypatch.setattr(
        scheduler, "_close_confirmed_jobs", lambda *args, **kwargs: _empty_async()
    )
    await scheduler.cleanup_theirstack_closed_jobs()
    assert [len(batch) for batch in calls] == [100, 100, 5]


@pytest.mark.asyncio
async def test_theirstack_cleanup_closes_only_explicit_valid_closed_jobs(monkeypatch):
    class Client:
        async def fetch_jobs_by_ids(self, ids):
            assert ids == ["1", "2", "3"]
            return [
                {"id": "1", "closed_at": "2026-10-06T00:00:00Z"},
                {"id": "2", "closed_at": None},
                {"id": "3", "closed_at": "not-a-date"},
            ]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def commit(self):
            pass

        async def execute(self, *_args, **_kwargs):
            return Result([("1",), ("2",), ("3",)])

    closed = []

    async def close_jobs(_db, *, ats_type, ats_job_ids):
        assert ats_type == "theirstack"
        closed.append(ats_job_ids)
        return ["db-1"]

    monkeypatch.setattr(scheduler, "_get_session_local", lambda: lambda: Session())
    monkeypatch.setattr(
        "app.job_ingestion.connectors.theirstack.TheirStackClient", Client
    )
    monkeypatch.setattr(scheduler, "_close_confirmed_jobs", close_jobs)
    monkeypatch.setattr(
        "app.job_ingestion.qdrant_service.delete_job_embedding", lambda _job_id: None
    )

    stats = await scheduler.cleanup_theirstack_closed_jobs()

    assert closed == [{"1"}]
    assert stats["jobs_checked"] == 3
    assert stats["jobs_confirmed_closed"] == 1


async def _empty_async():
    return []
