"""Add Playwright support columns

Revision ID: 002
Revises: 001
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add columns for Playwright scraping support."""
    
    # Add rendering_method column to track how content was scraped
    op.add_column(
        'articles',
        sa.Column(
            'rendering_method',
            sa.String(20),
            nullable=True,
            server_default='html',
            comment='Method used to render the page: html, playwright'
        )
    )
    
    # Add screenshot_path for storing screenshots
    op.add_column(
        'articles',
        sa.Column(
            'screenshot_path',
            sa.Text,
            nullable=True,
            comment='Path to screenshot if captured during scraping'
        )
    )
    
    # Add js_errors JSONB column for tracking JavaScript errors
    op.add_column(
        'articles',
        sa.Column(
            'js_errors',
            postgresql.JSONB,
            nullable=True,
            server_default='[]',
            comment='JavaScript errors encountered during page load'
        )
    )
    
    # Add page_metrics JSONB column for performance metrics
    op.add_column(
        'articles',
        sa.Column(
            'page_metrics',
            postgresql.JSONB,
            nullable=True,
            server_default='{}',
            comment='Page performance metrics (load time, resource count, etc.)'
        )
    )
    
    # Create index on rendering_method for performance
    op.create_index(
        'idx_articles_rendering_method',
        'articles',
        ['rendering_method'],
        postgresql_using='btree'
    )
    
    # Update metadata column to include Playwright-specific data
    op.execute("""
        COMMENT ON COLUMN articles.metadata IS 
        'Additional article metadata including Playwright scraping details'
    """)
    
    # Create a new table for tracking browser session stats
    op.create_table(
        'playwright_stats',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('session_id', postgresql.UUID, nullable=False, server_default=sa.text("uuid_generate_v4()")),
        sa.Column('browser_type', sa.String(20), nullable=False),
        sa.Column('pages_loaded', sa.Integer, server_default='0'),
        sa.Column('js_executed', sa.Integer, server_default='0'),
        sa.Column('errors', sa.Integer, server_default='0'),
        sa.Column('fallbacks', sa.Integer, server_default='0'),
        sa.Column('total_time_seconds', sa.Float, server_default='0.0'),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('CURRENT_TIMESTAMP'), 
                  onupdate=sa.text('CURRENT_TIMESTAMP'))
    )
    
    # Create index on session_id
    op.create_index(
        'idx_playwright_stats_session_id',
        'playwright_stats',
        ['session_id'],
        postgresql_using='btree'
    )
    
    # Add trigger to update updated_at
    op.execute("""
        CREATE TRIGGER update_playwright_stats_updated_at 
        BEFORE UPDATE ON playwright_stats
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
    """)
    
    # Update sources table to include js_required flag
    op.add_column(
        'sources',
        sa.Column(
            'js_required',
            sa.Boolean,
            nullable=False,
            server_default='false',
            comment='Whether this source requires JavaScript rendering'
        )
    )
    
    # Add playwright_settings to sources table
    op.add_column(
        'sources',
        sa.Column(
            'playwright_settings',
            postgresql.JSONB,
            nullable=True,
            server_default='{}',
            comment='Playwright-specific settings for this source'
        )
    )
    
    # Create a function to check if JavaScript is likely required
    op.execute("""
        CREATE OR REPLACE FUNCTION detect_js_requirement(html_content TEXT)
        RETURNS BOOLEAN AS $$
        DECLARE
            js_indicators TEXT[] := ARRAY[
                'react-root', 'ng-app', 'vue-app', '__NEXT_DATA__',
                'window.React', 'window.angular', 'window.Vue',
                'data-reactroot', 'data-ng-', 'v-app'
            ];
            indicator TEXT;
        BEGIN
            FOREACH indicator IN ARRAY js_indicators
            LOOP
                IF position(indicator IN html_content) > 0 THEN
                    RETURN true;
                END IF;
            END LOOP;
            RETURN false;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    
    # Create a view for Playwright scraping statistics
    op.execute("""
        CREATE VIEW playwright_scraping_stats AS
        SELECT 
            DATE(created_at) as date,
            browser_type,
            COUNT(*) as sessions,
            SUM(pages_loaded) as total_pages,
            SUM(js_executed) as total_js_executed,
            SUM(errors) as total_errors,
            SUM(fallbacks) as total_fallbacks,
            AVG(total_time_seconds / NULLIF(pages_loaded, 0)) as avg_time_per_page,
            SUM(errors)::float / NULLIF(SUM(pages_loaded), 0) as error_rate,
            SUM(fallbacks)::float / NULLIF(SUM(pages_loaded), 0) as fallback_rate
        FROM playwright_stats
        GROUP BY DATE(created_at), browser_type
        ORDER BY date DESC, browser_type;
    """)


def downgrade() -> None:
    """Remove Playwright support columns."""
    
    # Drop the view
    op.execute("DROP VIEW IF EXISTS playwright_scraping_stats")
    
    # Drop the function
    op.execute("DROP FUNCTION IF EXISTS detect_js_requirement")
    
    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS update_playwright_stats_updated_at ON playwright_stats")
    
    # Drop playwright_stats table
    op.drop_table('playwright_stats')
    
    # Remove columns from sources table
    op.drop_column('sources', 'playwright_settings')
    op.drop_column('sources', 'js_required')
    
    # Remove columns from articles table
    op.drop_index('idx_articles_rendering_method', table_name='articles')
    op.drop_column('articles', 'page_metrics')
    op.drop_column('articles', 'js_errors')
    op.drop_column('articles', 'screenshot_path')
    op.drop_column('articles', 'rendering_method')