"""add_ivfflat_index

Revision ID: ddbab9727dfc
Revises: cf35cf273f1d
Create Date: 2026-04-15 20:43:10.093990

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ddbab9727dfc'
down_revision: Union[str, None] = 'cf35cf273f1d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
        op.execute("""
        CREATE INDEX IF NOT EXISTS chunk_embedding_ivfflat_idx
        ON chunk
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunk_embedding_ivfflat_idx")
