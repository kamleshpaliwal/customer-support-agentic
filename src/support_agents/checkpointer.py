"""Durable graph checkpointer, created on the running application's event loop."""

from __future__ import annotations

from support_agents.settings import get_settings


async def create_checkpointer():
    """Return the configured checkpointer (PostgreSQL by default, memory for dev/tests).

    The PostgreSQL pool is bound to the event loop that creates it, so call this from the
    application's lifespan and close it with :func:`close_checkpointer` on shutdown.
    """
    settings = get_settings()
    if settings.checkpoint_backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    return await create_postgres_checkpointer(settings.database_url)


async def create_postgres_checkpointer(database_url: str):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    # autocommit and dict_row are required by AsyncPostgresSaver.
    pool = AsyncConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    # Explicitly allow our state types so strict msgpack mode does not block them.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("support_agents.schemas", name)
            for name in (
                "Intent",
                "FeeLookupRequest",
                "Evidence",
                "ToolCallRecord",
                "SpecialistResult",
            )
        ]
    )
    saver = AsyncPostgresSaver(pool, serde=serde)
    await saver.setup()  # creates the checkpoint tables if missing; idempotent
    return saver


async def close_checkpointer(checkpointer) -> None:
    conn = getattr(checkpointer, "conn", None)
    if conn is not None and hasattr(conn, "close"):
        await conn.close()
