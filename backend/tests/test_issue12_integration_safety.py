import pytest
from app.job_ingestion import qdrant_service
from tests.integration.issue12_config import (
    assert_safe_cleanup_collection,
    create_issue12_qdrant_client,
    delete_issue12_collection,
    require_issue12_database_url,
    require_issue12_qdrant_config,
)


def test_qdrant_collection_defaults_to_production_name(monkeypatch):
    monkeypatch.delenv("QDRANT_COLLECTION_NAME", raising=False)
    assert qdrant_service.DEFAULT_COLLECTION_NAME == "job_chunks"
    assert qdrant_service.qdrant_config_from_env({}).collection_name == "job_chunks"


def test_qdrant_collection_override_is_used_for_fresh_configuration():
    config = qdrant_service.qdrant_config_from_env(
        {"QDRANT_URL": "http://localhost:6333", "QDRANT_COLLECTION_NAME": "issue12_local_test"}
    )
    assert config.collection_name == "issue12_local_test"
    assert config.url == "http://localhost:6333"


def test_qdrant_client_factory_uses_explicit_isolated_configuration(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(qdrant_service, "QdrantClient", FakeClient)
    qdrant_service.create_qdrant_client(
        qdrant_service.QdrantConfig(
            url="http://localhost:6333",
            api_key=None,
            collection_name="issue12_local_test",
        )
    )
    assert calls == [{"url": "http://localhost:6333"}]


def test_qdrant_operations_use_explicit_client_collection():
    class Client:
        _eve_qdrant_collection = "issue12_local_test"

        def get_collections(self):
            return type("Result", (), {"collections": []})()

        def create_collection(self, **kwargs):
            self.created = kwargs

    client = Client()
    qdrant_service.ensure_collection(client=client)
    assert client.created["collection_name"] == "issue12_local_test"


def test_raw_custom_client_cannot_fall_back_to_production_collection():
    class Client:
        pass

    with pytest.raises(ValueError):
        qdrant_service.ensure_collection(client=Client())


def test_explicit_custom_collection_works_for_unconfigured_client():
    class Client:
        def get_collections(self):
            return type("Result", (), {"collections": []})()

        def create_collection(self, **kwargs):
            self.created = kwargs

    client = Client()
    qdrant_service.ensure_collection(client=client, collection_name="issue12_local_test")
    assert client.created["collection_name"] == "issue12_local_test"


def test_issue12_qdrant_config_requires_local_test_collection(monkeypatch):
    monkeypatch.setenv("ISSUE12_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("ISSUE12_QDRANT_COLLECTION", "issue12_local_test")
    assert require_issue12_qdrant_config() == (
        "http://localhost:6333",
        "issue12_local_test",
    )


@pytest.mark.parametrize(
    ("url", "collection"),
    [
        ("http://localhost:6333", "job_chunks"),
        ("http://localhost:6333", "not_a_test_collection"),
        ("https://qdrant.example.com", "issue12_remote_test"),
    ],
)
def test_issue12_qdrant_config_rejects_unsafe_targets(monkeypatch, url, collection):
    monkeypatch.setenv("ISSUE12_QDRANT_URL", url)
    monkeypatch.setenv("ISSUE12_QDRANT_COLLECTION", collection)
    with pytest.raises(RuntimeError):
        require_issue12_qdrant_config()


def test_issue12_database_requires_explicit_local_test_database(monkeypatch):
    monkeypatch.setenv(
        "ISSUE12_TEST_DATABASE_URL",
        "postgresql+asyncpg://user:pass@127.0.0.1:5432/pontis_issue12_test",
    )
    assert require_issue12_database_url().endswith("/pontis_issue12_test")


@pytest.mark.parametrize(
    "value",
    [
        "postgresql+asyncpg://user:pass@127.0.0.1:5432/pontis",
        "postgresql+asyncpg://user:pass@remote.example/pontis_issue12_test",
        "postgresql+asyncpg://user:pass@127.0.0.1:5432/pontis_production_test",
    ],
)
def test_issue12_database_rejects_unsafe_targets(monkeypatch, value):
    monkeypatch.setenv("ISSUE12_TEST_DATABASE_URL", value)
    with pytest.raises(RuntimeError):
        require_issue12_database_url()


def test_issue12_cleanup_is_restricted_to_configured_collection(monkeypatch):
    monkeypatch.setenv("ISSUE12_QDRANT_COLLECTION", "issue12_local_test")
    assert_safe_cleanup_collection("issue12_local_test")
    with pytest.raises(RuntimeError):
        assert_safe_cleanup_collection("job_chunks")
    with pytest.raises(RuntimeError):
        assert_safe_cleanup_collection("issue12_other_test")


def test_issue12_collection_cleanup_uses_only_validated_collection(monkeypatch):
    monkeypatch.setenv("ISSUE12_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("ISSUE12_QDRANT_COLLECTION", "issue12_local_test")
    deleted = []

    monkeypatch.setattr(qdrant_service, "QdrantClient", lambda **_kwargs: type("Client", (), {})())
    client = create_issue12_qdrant_client()

    def delete_collection(self, *, collection_name):
        deleted.append(collection_name)

    client.delete_collection = delete_collection.__get__(client)

    delete_issue12_collection(client)
    assert deleted == ["issue12_local_test"]


def test_issue12_cleanup_rejects_client_for_different_endpoint(monkeypatch):
    monkeypatch.setenv("ISSUE12_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("ISSUE12_QDRANT_COLLECTION", "issue12_local_test")

    class Client:
        def delete_collection(self, **_kwargs):
            raise AssertionError("unsafe cleanup must not call the client")

    with pytest.raises(RuntimeError):
        delete_issue12_collection(Client())


def test_issue12_cleanup_rejects_arbitrary_client_with_matching_attributes(monkeypatch):
    monkeypatch.setenv("ISSUE12_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("ISSUE12_QDRANT_COLLECTION", "issue12_local_test")

    class Client:
        _eve_qdrant_url = "http://localhost:6333"
        _eve_qdrant_collection = "issue12_local_test"

    with pytest.raises(RuntimeError):
        delete_issue12_collection(Client())
