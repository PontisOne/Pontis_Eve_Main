"""Short-lived PostgreSQL coordination primitives for recommendation refreshes.

These functions deliberately accept an existing AsyncSession-like object.  Each
operation performs one short SQL statement; the caller owns the transaction and
must commit or roll back before doing any retrieval or scoring work.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text


def _validate_lease_duration(lease_duration_seconds: int) -> None:
    if lease_duration_seconds <= 0:
        raise ValueError("lease_duration_seconds must be greater than zero")


@dataclass(frozen=True)
class RefreshLeaseAcquisition:
    acquired: bool
    generation: int | None = None
    owner_token: str | None = None
    lease_expires_at: Any = None


async def acquire_refresh_lease(
    db: Any,
    candidate_id: str,
    owner_token: str,
    lease_duration_seconds: int,
) -> RefreshLeaseAcquisition:
    """Atomically acquire or reacquire an expired candidate lease.

    A successful acquisition increments generation.  A rejected acquisition
    returns ``acquired=False`` and does not modify the row.  The caller must
    commit this short transaction before beginning refresh computation.
    """
    _validate_lease_duration(lease_duration_seconds)
    result = await db.execute(
        text(
            """
            INSERT INTO candidate_recommendation_refresh_leases
                (candidate_id, generation, owner_token, lease_expires_at, updated_at)
            VALUES
                (:candidate_id, 1, CAST(:owner_token AS uuid),
                 now() + (:lease_duration_seconds * INTERVAL '1 second'), now())
            ON CONFLICT (candidate_id) DO UPDATE
            SET generation = candidate_recommendation_refresh_leases.generation + 1,
                owner_token = EXCLUDED.owner_token,
                lease_expires_at = EXCLUDED.lease_expires_at,
                updated_at = now()
            WHERE candidate_recommendation_refresh_leases.owner_token IS NULL
               OR candidate_recommendation_refresh_leases.lease_expires_at IS NULL
               OR candidate_recommendation_refresh_leases.lease_expires_at <= now()
            RETURNING generation, owner_token, lease_expires_at
            """
        ),
        {
            "candidate_id": candidate_id,
            "owner_token": owner_token,
            "lease_duration_seconds": lease_duration_seconds,
        },
    )
    row = result.mappings().fetchone()
    if not row:
        return RefreshLeaseAcquisition(acquired=False)
    return RefreshLeaseAcquisition(
        acquired=True,
        generation=int(row["generation"]),
        owner_token=str(row["owner_token"]),
        lease_expires_at=row["lease_expires_at"],
    )


async def renew_refresh_lease(
    db: Any,
    candidate_id: str,
    owner_token: str,
    generation: int,
    lease_duration_seconds: int,
) -> bool:
    """Renew only the currently owned, unexpired generation."""
    _validate_lease_duration(lease_duration_seconds)
    result = await db.execute(
        text(
            """
            UPDATE candidate_recommendation_refresh_leases
            SET lease_expires_at = now() + (
                :lease_duration_seconds * INTERVAL '1 second'
            ),
                updated_at = now()
            WHERE candidate_id = :candidate_id
              AND owner_token = CAST(:owner_token AS uuid)
              AND generation = :generation
              AND lease_expires_at > now()
            """
        ),
        {
            "candidate_id": candidate_id,
            "owner_token": owner_token,
            "generation": generation,
            "lease_duration_seconds": lease_duration_seconds,
        },
    )
    return result.rowcount == 1


async def release_refresh_lease(
    db: Any,
    candidate_id: str,
    owner_token: str,
    generation: int,
) -> bool:
    """Release only the exact owner and fencing generation."""
    result = await db.execute(
        text(
            """
            UPDATE candidate_recommendation_refresh_leases
            SET owner_token = NULL,
                lease_expires_at = NULL,
                updated_at = now()
            WHERE candidate_id = :candidate_id
              AND owner_token = CAST(:owner_token AS uuid)
              AND generation = :generation
            """
        ),
        {
            "candidate_id": candidate_id,
            "owner_token": owner_token,
            "generation": generation,
        },
    )
    return result.rowcount == 1
