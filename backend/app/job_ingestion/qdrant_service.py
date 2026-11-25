import os
import uuid
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from location_matching import country_code
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from skill_normalization import canonical_skill, merge_skills

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_VERSION = os.getenv("EMBEDDING_VERSION", "v2_structured")

DEFAULT_COLLECTION_NAME = "job_chunks"
VECTOR_SIZE = 384


@dataclass(frozen=True)
class QdrantConfig:
    """Import-time Qdrant configuration, overridable before module import."""

    url: str | None
    api_key: str | None
    collection_name: str


def qdrant_config_from_env(environ=None) -> QdrantConfig:
    values = os.environ if environ is None else environ
    return QdrantConfig(
        url=values.get("QDRANT_URL"),
        api_key=values.get("QDRANT_API_KEY"),
        collection_name=values.get("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
    )


_CONFIG = qdrant_config_from_env()
QDRANT_URL = _CONFIG.url
QDRANT_API_KEY = _CONFIG.api_key
COLLECTION_NAME = _CONFIG.collection_name


def get_qdrant_config() -> QdrantConfig:
    """Return the effective Qdrant settings without exposing credentials."""
    return _CONFIG


def create_qdrant_client(config: QdrantConfig | None = None) -> QdrantClient:
    """Create a client from explicit or import-time configuration."""
    effective = config or get_qdrant_config()
    kwargs: dict[str, Any] = {"url": effective.url}
    if effective.api_key:
        kwargs["api_key"] = effective.api_key
    client = QdrantClient(**kwargs)
    url_attribute = "_eve_qdrant_url"
    collection_attribute = "_eve_qdrant_collection"
    setattr(client, url_attribute, effective.url)
    setattr(client, collection_attribute, effective.collection_name)
    return client


_client = create_qdrant_client()


def _target(client=None, collection_name=None):
    effective_client = client or _client
    configured_collection = getattr(effective_client, "_eve_qdrant_collection", None)
    if (
        effective_client is not _client
        and not configured_collection
        and not collection_name
    ):
        raise ValueError("Custom Qdrant clients require an explicit collection")
    effective_collection = collection_name or configured_collection or COLLECTION_NAME
    if (
        configured_collection
        and collection_name
        and configured_collection != collection_name
    ):
        raise ValueError("Qdrant client and collection configuration do not match")
    return effective_client, effective_collection


def ensure_collection(*, client=None, collection_name=None) -> None:
    target_client, target_collection = _target(client, collection_name)
    existing = {c.name for c in target_client.get_collections().collections}
    if target_collection not in existing:
        target_client.create_collection(
            collection_name=target_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _point_id(job_id: str) -> int:
    return uuid.UUID(job_id).int % (2**63)


def delete_job_embedding(job_id: str, *, client=None, collection_name=None) -> bool:
    """Delete exactly this job's deterministic point, never a broad filter."""
    from qdrant_client.models import PointIdsList

    target_client, target_collection = _target(client, collection_name)
    target_client.delete(
        collection_name=target_collection,
        points_selector=PointIdsList(points=[_point_id(job_id)]),
    )
    return True


def _existing_payload(job_id: str, *, client=None, collection_name=None) -> dict | None:
    target_client, target_collection = _target(client, collection_name)
    points = target_client.retrieve(
        collection_name=target_collection,
        ids=[_point_id(job_id)],
        with_payload=True,
        with_vectors=False,
    )
    return dict(points[0].payload or {}) if points else None


def upsert_job_embedding(
    job_id: str,
    embedding: list[float],
    job: dict,
    *,
    reindex: bool = False,
    client=None,
    collection_name=None,
) -> bool:
    """Upsert a stable job point, requiring explicit replacement of incompatible vectors."""
    if len(embedding) != VECTOR_SIZE:
        raise ValueError(
            f"Expected {VECTOR_SIZE}-dimensional job embedding, got {len(embedding)}"
        )
    target_client, target_collection = _target(client, collection_name)
    existing = _existing_payload(
        job_id, client=target_client, collection_name=target_collection
    )
    if existing:
        existing_version = existing.get("embedding_version")
        existing_model = existing.get("embedding_model")
        incompatible = (existing_version and existing_version != EMBEDDING_VERSION) or (
            existing_model and existing_model != EMBEDDING_MODEL_NAME
        )
        if incompatible and not reindex:
            raise ValueError(
                "Existing point has incompatible embedding metadata "
                f"(version={existing_version!r}, model={existing_model!r}); "
                "rerun with reindex=True to replace it explicitly"
            )

    point_id = _point_id(job_id)
    target_client.upsert(
        collection_name=target_collection,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "job_id": job_id,
                    "ats_job_id": job.get("ats_job_id"),
                    "title": job.get("title"),
                    "company_name": job.get("company_name"),
                    "ats_type": job.get("ats_type"),
                    "country": job.get("country"),
                    "country_code": country_code(job.get("country")),
                    "city": job.get("city"),
                    "state": job.get("state"),
                    "location": job.get("location"),
                    "remote": job.get("remote"),
                    "skills": merge_skills(
                        job.get("skills_required"), job.get("skills")
                    ),
                    "canonical_skills": [
                        canonical_skill(value)
                        for value in merge_skills(
                            job.get("skills_required"), job.get("skills")
                        )
                    ],
                    "embedding_version": EMBEDDING_VERSION,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                },
            )
        ],
    )
    return True
