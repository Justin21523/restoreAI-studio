from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from restorai.settings import get_settings


class Base(DeclarativeBase):
    pass


SCHEMA_REVISION = "20260710_0002"


@lru_cache(maxsize=4)
def get_engine(database_url: str | None = None):  # type: ignore[no-untyped-def]
    url = database_url or get_settings().database_url
    if url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
    elif url.startswith("postgresql+psycopg"):
        # RQ forks a short-lived child for every GPU job. Psycopg's automatic
        # server-side prepare cache is process-local and can reuse statement names
        # after a fork, producing DuplicatePreparedStatement on later jobs.
        kwargs = {"connect_args": {"prepare_threshold": None}}
    else:
        kwargs = {}
    return create_engine(url, pool_pre_ping=True, **kwargs)


def session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)


def require_database_schema(database_url: str | None = None) -> None:
    """Fail fast when Alembic has not installed the expected schema."""
    try:
        with get_engine(database_url).connect() as connection:
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
    except SQLAlchemyError as exc:
        raise RuntimeError("Database schema is missing; run `alembic upgrade head`") from exc
    if revision != SCHEMA_REVISION:
        raise RuntimeError(
            f"Database schema is {revision!r}, expected {SCHEMA_REVISION!r}; "
            "run `alembic upgrade head`"
        )


def get_session() -> Generator[Session, None, None]:
    factory = session_factory()
    with factory() as session:
        yield session
