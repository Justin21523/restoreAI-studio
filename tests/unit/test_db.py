from __future__ import annotations

from pathlib import Path

import pytest

from restorai.db import require_database_schema


def test_schema_check_requires_alembic_revision(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="alembic upgrade head"):
        require_database_schema(f"sqlite:///{tmp_path / 'empty.db'}")
