from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from app.services.voice_intake_persistence_service import (
    VoiceIntakePersistenceDependencies,
    VoiceIntakePersistenceService,
)


class _Cache:
    def __init__(self) -> None:
        self.invalidate = AsyncMock()


def _session_factory(db):
    @asynccontextmanager
    async def factory():
        yield db

    return factory


@pytest.mark.asyncio
async def test_persistence_services_keep_session_and_cache_dependencies_isolated():
    db_one = AsyncMock()
    db_two = AsyncMock()
    cache_one = _Cache()
    cache_two = _Cache()

    service_one = VoiceIntakePersistenceService(
        VoiceIntakePersistenceDependencies(
            session_local=_session_factory(db_one),
            parse_raw_data=lambda value: value,
            profile_cache=lambda: cache_one,
        )
    )
    service_two = VoiceIntakePersistenceService(
        VoiceIntakePersistenceDependencies(
            session_local=_session_factory(db_two),
            parse_raw_data=lambda value: value,
            profile_cache=lambda: cache_two,
        )
    )

    await service_one.save_voice_intake_resume("candidate-one", {"status": "done"})
    await service_two.save_voice_intake_resume("candidate-two", {"status": "done"})

    db_one.execute.assert_awaited_once()
    db_two.execute.assert_awaited_once()
    db_one.commit.assert_awaited_once()
    db_two.commit.assert_awaited_once()
    cache_one.invalidate.assert_awaited_once_with("candidate-one")
    cache_two.invalidate.assert_awaited_once_with("candidate-two")
