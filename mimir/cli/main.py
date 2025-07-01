"""Main CLI runner for Mimir."""

import asyncio
import logging
from typing import Optional

import click
import uvicorn

from ..config import get_settings
from ..core.database import DatabaseManager
from ..core.services import ScrapingService
from ..core.database import get_async_session


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
def cli(debug: bool):
    """Mimir Broadcasting Intelligence Platform CLI."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--workers', default=1, help='Number of worker processes')
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the API server."""
    click.echo(f"Starting Mimir API server on {host}:{port}")
    
    if reload:
        # Development mode with reload
        uvicorn.run(
            "mimir.api.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    else:
        # Production mode
        uvicorn.run(
            "mimir.api.app:app",
            host=host,
            port=port,
            workers=workers,
            log_level="warning"
        )


@cli.command()
@click.option('--enabled-only', is_flag=True, default=True, help='Only scrape enabled sources')
@click.option('--max-articles', type=int, help='Maximum articles per source')
async def scrape(enabled_only: bool, max_articles: Optional[int]):
    """Run scraping operation."""
    click.echo("Starting scraping operation...")
    
    try:
        # Initialize database
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Run scraping
        async with get_async_session() as session:
            scraping_service = ScrapingService(session)
            result = await scraping_service.scrape_all_sources(
                enabled_only=enabled_only,
                max_articles_per_source=max_articles
            )
        
        # Display results
        click.echo(f"✅ Scraping completed!")
        click.echo(f"   Sources processed: {result['sources_processed']}")
        click.echo(f"   Total articles found: {result['total_articles']}")
        click.echo(f"   New articles saved: {result['new_articles']}")
        
        if result['errors']:
            click.echo(f"   ⚠️  Errors: {len(result['errors'])}")
            for error in result['errors'][:5]:  # Show first 5 errors
                click.echo(f"      - {error}")
    
    except Exception as e:
        click.echo(f"❌ Scraping failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
async def init_db():
    """Initialize the database."""
    click.echo("Initializing database...")
    
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        click.echo("✅ Database initialized successfully!")
    
    except Exception as e:
        click.echo(f"❌ Database initialization failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
async def health():
    """Check system health."""
    click.echo("Checking system health...")
    
    try:
        # Test database connection
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        async with get_async_session() as session:
            await session.execute("SELECT 1")
        
        click.echo("✅ Database: Connected")
        
        # Test configuration
        settings = get_settings()
        click.echo(f"✅ Configuration: Loaded (debug={settings.debug})")
        
        click.echo("🟢 System is healthy!")
    
    except Exception as e:
        click.echo(f"❌ Health check failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
def config():
    """Show current configuration."""
    settings = get_settings()
    
    click.echo("Current Configuration:")
    click.echo("=" * 40)
    click.echo(f"Debug Mode: {settings.debug}")
    click.echo(f"Database URL: {settings.database.url}")
    click.echo(f"Max Articles per Source: {settings.scraping.articles_per_source}")
    click.echo(f"Request Timeout: {settings.scraping.request_timeout}s")
    click.echo(f"Keywords: {', '.join(settings.monitoring.keywords)}")


# Async command wrapper
def async_command(func):
    """Decorator to run async commands."""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


# Apply async wrapper to async commands
scrape = async_command(scrape)
init_db = async_command(init_db)
health = async_command(health)


if __name__ == '__main__':
    cli()