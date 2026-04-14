"""add ivfflat index on chunk embedding

Revision ID: 6a3f7c1b6d37
Revises: d7c2a231ca4d
Create Date: 2026-04-07 06:56:49.123770

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6a3f7c1b6d37'
down_revision: Union[str, None] = 'd7c2a231ca4d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunk_embedding_ivfflat_idx
        ON chunk
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10)
    """)
    # lists=10 because you have small data now
    # increase to 100 when you have 10k+ rows

def downgrade():
    op.execute("DROP INDEX IF EXISTS chunk_embedding_ivfflat_idx")
