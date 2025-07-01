"""Initial PostgreSQL schema creation

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """TODO: Add docstring."""
    # Execute the schema creation SQL
    with open("scripts/postgres/init/01_schema.sql", "r") as f:
        schema_sql = f.read()

    # Split by ';' and execute each statement
    statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
    for statement in statements:
        if statement:
            op.execute(statement + ";")

    # Execute seed data
    with open("scripts/postgres/init/02_seed_data.sql", "r") as f:
        seed_sql = f.read()

    statements = [s.strip() for s in seed_sql.split(";") if s.strip()]
    for statement in statements:
        if statement and not statement.startswith("--"):
            try:
                op.execute(statement + ";")
            except Exception as e:
                # Log but don't fail on seed data errors
                print(f"Warning: Seed data statement failed: {e}")


def downgrade() -> None:
    """TODO: Add docstring."""
    # Drop all tables in reverse order of dependencies
    op.drop_table("article_embeddings")
    op.drop_table("notifications")
    op.drop_table("keyword_alerts")
    op.drop_table("scrape_errors")
    op.drop_table("scrape_runs")
    op.drop_table("keywords")
    op.drop_table("sources")
    op.drop_table("articles")
    op.drop_table("users")

    # Drop views and functions
    op.execute("DROP VIEW IF EXISTS recent_articles_with_alerts CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS article_stats CASCADE")
    op.execute("DROP FUNCTION IF EXISTS search_articles CASCADE")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column CASCADE")
    op.execute("DROP FUNCTION IF EXISTS calculate_next_scrape_time CASCADE")
    op.execute("DROP FUNCTION IF EXISTS import_keywords_from_config CASCADE")

    # Drop types
    op.execute("DROP TYPE IF EXISTS article_status CASCADE")
    op.execute("DROP TYPE IF EXISTS scrape_type CASCADE")

    # Drop extensions
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp" CASCADE')
    op.execute('DROP EXTENSION IF EXISTS "pg_trgm" CASCADE')
    op.execute('DROP EXTENSION IF EXISTS "btree_gin" CASCADE')
