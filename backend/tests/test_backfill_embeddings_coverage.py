import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def fetchall(self):
        return self.rows

    def scalar(self):
        return len(self.rows)


class _Session:
    def __init__(self, jobs):
        self.jobs = jobs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, statement, params=None):
        # The test data includes a non-ATS job: it must be selected as visible.
        assert "WHERE ats_type" not in str(statement)
        assert "is_active IS TRUE" in str(statement)
        return _Rows(self.jobs)


def _record_embedding(embedded, job):
    embedded.append(job)
    return [0.1] * 384


def test_backfill_selects_candidate_visible_non_ats_jobs(monkeypatch):
    jobs = [
        {
            "id": "non-ats-job",
            "ats_type": "custom",
            "title": "Java Engineer",
            "company_name": "Acme",
            "job_url": "https://jobs.example/non-ats-job",
            "description": "Build Java services.",
        }
    ]
    server = ModuleType("server")
    server.__dict__["SessionLocal"] = lambda: _Session(jobs)
    sqlalchemy = ModuleType("sqlalchemy")
    sqlalchemy.__dict__["text"] = lambda value: value
    embedding = ModuleType("app.job_ingestion.embedding_service")
    embedding.__dict__["generate_job_embedding"] = lambda job: [0.1] * 384
    qdrant = ModuleType("app.job_ingestion.qdrant_service")
    written = []
    embedded: list[dict[str, Any]] = []
    qdrant.__dict__["ensure_collection"] = lambda: None
    qdrant.__dict__["delete_job_embedding"] = lambda *_args: True
    embedding.__dict__["generate_job_embedding"] = lambda job: _record_embedding(embedded, job)
    qdrant.__dict__["upsert_job_embedding"] = lambda job_id, vector, job, **kwargs: written.append((job_id, vector, job))

    monkeypatch.setitem(sys.modules, "server", server)
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.embedding_service", embedding)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.qdrant_service", qdrant)
    path = (
        Path(__file__).parents[1] / "app" / "job_ingestion" / "backfill_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location("backfill_embeddings_coverage", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load backfill module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    asyncio.run(module.backfill())
    assert len(embedded) == 1
    assert written == [("non-ats-job", [0.1] * 384, embedded[0])]


def test_backfill_deletes_ineligible_points_and_does_not_embed_them(monkeypatch):
    jobs = [
        {
            "id": "eligible",
            "ats_type": "custom",
            "title": "Engineer",
            "company_name": "Acme",
            "job_url": "https://jobs.example/eligible",
            "description": "Build APIs.",
        },
        {
            "id": "ineligible",
            "ats_type": "custom",
            "title": "Engineer",
            "company_name": "Acme",
            "job_url": "https://jobs.example/ineligible",
            "description": "",
        },
    ]
    server = ModuleType("server")
    server.__dict__["SessionLocal"] = lambda: _Session(jobs)
    sqlalchemy = ModuleType("sqlalchemy")
    sqlalchemy.__dict__["text"] = lambda value: value
    embedding = ModuleType("app.job_ingestion.embedding_service")
    qdrant = ModuleType("app.job_ingestion.qdrant_service")
    embedded: list[dict[str, Any]] = []
    upserted, deleted = [], []
    embedding.__dict__["generate_job_embedding"] = lambda job: _record_embedding(embedded, job)
    qdrant.__dict__["ensure_collection"] = lambda: None
    qdrant.__dict__["upsert_job_embedding"] = lambda job_id, vector, job, **kwargs: upserted.append((job_id, vector, job))
    qdrant.__dict__["delete_job_embedding"] = lambda job_id: deleted.append(job_id)
    monkeypatch.setitem(sys.modules, "server", server)
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.embedding_service", embedding)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.qdrant_service", qdrant)
    path = Path(__file__).parents[1] / "app" / "job_ingestion" / "backfill_embeddings.py"
    spec = importlib.util.spec_from_file_location("backfill_embeddings_quality", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load backfill module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    asyncio.run(module.backfill(reindex=True))

    assert [job["id"] for job in embedded] == ["eligible"]
    assert [item[0] for item in upserted] == ["eligible"]
    assert deleted == ["ineligible"]


def test_backfill_deletion_failure_is_counted_without_aborting(monkeypatch, caplog, capsys):
    jobs = [
        {
            "id": "ineligible",
            "ats_type": "custom",
            "title": "Engineer",
            "company_name": "Acme",
            "job_url": "https://jobs.example/ineligible",
            "description": "",
        },
        {
            "id": "eligible-after-failure",
            "ats_type": "custom",
            "title": "Engineer",
            "company_name": "Acme",
            "job_url": "https://jobs.example/eligible-after-failure",
            "description": "Build APIs.",
        },
    ]
    server = ModuleType("server")
    server.__dict__["SessionLocal"] = lambda: _Session(jobs)
    sqlalchemy = ModuleType("sqlalchemy")
    sqlalchemy.__dict__["text"] = lambda value: value
    embedding = ModuleType("app.job_ingestion.embedding_service")
    qdrant = ModuleType("app.job_ingestion.qdrant_service")
    embedded: list[dict[str, Any]] = []
    upserted = []
    embedding.__dict__["generate_job_embedding"] = lambda job: _record_embedding(embedded, job)
    qdrant.__dict__["ensure_collection"] = lambda: None
    qdrant.__dict__["upsert_job_embedding"] = lambda job_id, vector, job, **_kwargs: upserted.append((job_id, vector, job))
    qdrant.__dict__["delete_job_embedding"] = lambda *_args: (_ for _ in ()).throw(RuntimeError("delete failed"))
    monkeypatch.setitem(sys.modules, "server", server)
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.embedding_service", embedding)
    monkeypatch.setitem(sys.modules, "app.job_ingestion.qdrant_service", qdrant)
    path = Path(__file__).parents[1] / "app" / "job_ingestion" / "backfill_embeddings.py"
    spec = importlib.util.spec_from_file_location("backfill_embeddings_delete_failure", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load backfill module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with caplog.at_level("ERROR"):
        asyncio.run(module.backfill())

    assert [item[0] for item in upserted] == ["eligible-after-failure"]
    assert [job["id"] for job in embedded] == ["eligible-after-failure"]
    assert "failed job_id=ineligible" in caplog.text
    assert "succeeded=1 failed=1" in capsys.readouterr().out
