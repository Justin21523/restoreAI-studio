"""Add explicit batch retry lineage."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260710_0002"
down_revision = "20260710_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("batches", sa.Column("retry_of_batch_id", sa.String(36), nullable=True))
    op.create_foreign_key(
        "fk_batches_retry_of_batch_id",
        "batches",
        "batches",
        ["retry_of_batch_id"],
        ["id"],
    )
    op.create_index("ix_batches_retry_of_batch_id", "batches", ["retry_of_batch_id"])


def downgrade() -> None:
    op.drop_index("ix_batches_retry_of_batch_id", table_name="batches")
    op.drop_constraint("fk_batches_retry_of_batch_id", "batches", type_="foreignkey")
    op.drop_column("batches", "retry_of_batch_id")
