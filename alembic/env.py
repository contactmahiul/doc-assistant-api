from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, create_engine
import sqlalchemy as sa
from alembic import context
from alembic.autogenerate import rewriter
from alembic.operations import ops as alembic_ops

from app.db.base import Base
import os
from app.core.config import settings

# ── Alembic Config ────────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ── Guard 1: prevent autogenerate dropping GIN index ─────────────────────────
def include_object(object, name, type_, reflected, compare_to):
    if type_ == "index" and name == "idx_chunk_fts":
        return False
    return True


# ── Guard 2: prevent autogenerate modifying computed columns ──────────────────
writer = rewriter.Rewriter()

@writer.rewrites(alembic_ops.ModifyTableOps)
def ignore_computed_column_modifications(context, revision, directives):
    for directive in directives:
        directive.ops = [
            op for op in directive.ops
            if not (
                hasattr(op, "column")
                and isinstance(getattr(op.column, "computed", None), sa.Computed)
            )
        ]
    return directives


# ── Offline mode ──────────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,          # ✅ Guard 1
        process_revision_directives=writer,     # ✅ Guard 2
    )

    with context.begin_transaction():
        context.run_migrations()


# ── Online mode ───────────────────────────────────────────────────────────────
def run_migrations_online() -> None:
    db_url = settings.DATABASE_URL
    print(f"DIRECT OS URL: {db_url}", flush=True)

    connectable = create_engine(db_url)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,          # ✅ Guard 1
            process_revision_directives=writer,     # ✅ Guard 2
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()