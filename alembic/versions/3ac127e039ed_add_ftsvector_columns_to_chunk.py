"""Add ftsvector columns to chunk

Revision ID: 3ac127e039ed
Revises: 9c5192f3132e
Create Date: 2026-04-30 14:19:13.877580

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR
# revision identifiers, used by Alembic.
revision: str = '3ac127e039ed'
down_revision: Union[str, None] = '9c5192f3132e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chunk",
        sa.Column(
            "fts_vector",
            TSVECTOR,
            sa.Computed("to_tsvector('english', coalesce(content, ''))", persisted=True),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_chunk_fts",
        "chunk",
        ["fts_vector"],
        postgresql_using="gin",
    )

def downgrade() -> None:
    op.drop_index("idx_chunk_fts", table_name="chunk")
    op.drop_column("chunk", "fts_vector")
