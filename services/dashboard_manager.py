"""
Dashboard Manager Service for Mimir Enterprise

Manages custom dashboards and widgets with real-time data updates.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ..api.models.dashboard import (
    DashboardLayout,
    DashboardTheme,
    RefreshInterval,
    WidgetConfig,
    WidgetSize,
    WidgetType,
)

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration"""

    id: str
    name: str
    description: Optional[str]
    layout: DashboardLayout
    theme: DashboardTheme
    is_default: bool
    is_public: bool
    tags: List[str]
    settings: Dict[str, Any]
    organization_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime


@dataclass
class Widget:
    """Widget configuration"""

    id: str
    dashboard_id: str
    type: WidgetType
    title: str
    size: WidgetSize
    refresh_interval: RefreshInterval
    data_source: Dict[str, Any]
    display_options: Dict[str, Any]
    position: Dict[str, int]
    created_at: datetime
    updated_at: datetime


class DashboardManager:
    """Manages dashboards and widgets"""

    def __init__(self, db_manager, cache_manager=None):
        """
        Initialize dashboard manager

        Args:
            db_manager: Database manager instance
            cache_manager: Optional cache manager for widget data
        """
        self.db = db_manager
        self.cache = cache_manager

    def create_dashboard(
        self,
        organization_id: str,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        layout: DashboardLayout = DashboardLayout.GRID,
        theme: DashboardTheme = DashboardTheme.LIGHT,
        is_default: bool = False,
        is_public: bool = False,
        tags: List[str] = None,
        widgets: List[WidgetConfig] = None,
        settings: Dict[str, Any] = None,
    ) -> DashboardConfig:
        """
        Create a new dashboard

        Args:
            organization_id: Organization ID
            user_id: User ID
            name: Dashboard name
            description: Optional description
            layout: Layout type
            theme: Color theme
            is_default: Set as default dashboard
            is_public: Make publicly accessible
            tags: Dashboard tags
            widgets: Initial widgets
            settings: Dashboard settings

        Returns:
            Created dashboard configuration
        """
        try:
            dashboard_id = str(uuid4())

            # Create dashboard
            query = """
            INSERT INTO dashboards (
                id, organization_id, user_id, name, description,
                layout, theme, is_default, is_public, tags, settings
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """

            result = self.db.fetch_one(
                query,
                (
                    dashboard_id,
                    organization_id,
                    user_id,
                    name,
                    description,
                    layout,
                    theme,
                    is_default,
                    is_public,
                    json.dumps(tags or []),
                    json.dumps(settings or {}),
                ),
            )

            # Add initial widgets if provided
            if widgets:
                for widget_config in widgets:
                    self.add_widget_to_dashboard(
                        dashboard_id,
                        widget_config.type,
                        widget_config.title,
                        widget_config.size,
                        widget_config.refresh_interval,
                        widget_config.data_source,
                        widget_config.display_options,
                        widget_config.position,
                    )

            return DashboardConfig(
                id=result["id"],
                name=result["name"],
                description=result["description"],
                layout=result["layout"],
                theme=result["theme"],
                is_default=result["is_default"],
                is_public=result["is_public"],
                tags=result["tags"],
                settings=result["settings"],
                organization_id=result["organization_id"],
                user_id=result["user_id"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )

        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise

    def get_dashboard(
        self, dashboard_id: str, user_id: str, include_widgets: bool = True
    ) -> Optional[Tuple[DashboardConfig, List[Widget]]]:
        """
        Get dashboard by ID

        Args:
            dashboard_id: Dashboard ID
            user_id: User ID for access check
            include_widgets: Include widgets in response

        Returns:
            Tuple of (dashboard, widgets) or None
        """
        try:
            # Get dashboard
            query = """
            SELECT d.* FROM dashboards d
            LEFT JOIN dashboard_shares ds ON d.id = ds.dashboard_id
            WHERE d.id = %s
              AND (
                d.user_id = %s
                OR ds.shared_with_user_id = %s
                OR d.is_public = true
              )
            """

            result = self.db.fetch_one(query, (dashboard_id, user_id, user_id))

            if not result:
                return None

            dashboard = DashboardConfig(
                id=result["id"],
                name=result["name"],
                description=result["description"],
                layout=result["layout"],
                theme=result["theme"],
                is_default=result["is_default"],
                is_public=result["is_public"],
                tags=result["tags"],
                settings=result["settings"],
                organization_id=result["organization_id"],
                user_id=result["user_id"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )

            widgets = []
            if include_widgets:
                widgets = self.get_dashboard_widgets(dashboard_id)

            return dashboard, widgets

        except Exception as e:
            logger.error(f"Error getting dashboard: {e}")
            return None

    def list_user_dashboards(
        self, organization_id: str, user_id: str, include_shared: bool = True
    ) -> List[DashboardConfig]:
        """
        List dashboards accessible to user

        Args:
            organization_id: Organization ID
            user_id: User ID
            include_shared: Include shared dashboards

        Returns:
            List of dashboards
        """
        try:
            if include_shared:
                query = """
                SELECT DISTINCT d.* FROM dashboards d
                LEFT JOIN dashboard_shares ds ON d.id = ds.dashboard_id
                WHERE d.organization_id = %s
                  AND (
                    d.user_id = %s
                    OR ds.shared_with_user_id = %s
                    OR d.is_public = true
                  )
                ORDER BY d.is_default DESC, d.updated_at DESC
                """
                params = (organization_id, user_id, user_id)
            else:
                query = """
                SELECT * FROM dashboards
                WHERE organization_id = %s AND user_id = %s
                ORDER BY is_default DESC, updated_at DESC
                """
                params = (organization_id, user_id)

            results = self.db.fetch_all(query, params)

            return [
                DashboardConfig(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    layout=row["layout"],
                    theme=row["theme"],
                    is_default=row["is_default"],
                    is_public=row["is_public"],
                    tags=row["tags"],
                    settings=row["settings"],
                    organization_id=row["organization_id"],
                    user_id=row["user_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error listing dashboards: {e}")
            return []

    def update_dashboard(self, dashboard_id: str, user_id: str, **updates) -> Optional[DashboardConfig]:
        """
        Update dashboard

        Args:
            dashboard_id: Dashboard ID
            user_id: User ID for ownership check
            **updates: Fields to update

        Returns:
            Updated dashboard or None
        """
        try:
            # Build update query
            allowed_fields = ["name", "description", "layout", "theme", "is_default", "is_public", "tags", "settings"]

            update_fields = []
            params = []

            for field, value in updates.items():
                if field in allowed_fields:
                    update_fields.append(f"{field} = %s")
                    if field in ["tags", "settings"]:
                        params.append(json.dumps(value))
                    else:
                        params.append(value)

            if not update_fields:
                return None

            params.extend([dashboard_id, user_id])

            query = """
            UPDATE dashboards
            SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            RETURNING *
            """

            result = self.db.fetch_one(query, params)

            if not result:
                return None

            return DashboardConfig(
                id=result["id"],
                name=result["name"],
                description=result["description"],
                layout=result["layout"],
                theme=result["theme"],
                is_default=result["is_default"],
                is_public=result["is_public"],
                tags=result["tags"],
                settings=result["settings"],
                organization_id=result["organization_id"],
                user_id=result["user_id"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )

        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            return None

    def delete_dashboard(self, dashboard_id: str, user_id: str) -> bool:
        """
        Delete dashboard

        Args:
            dashboard_id: Dashboard ID
            user_id: User ID for ownership check

        Returns:
            True if deleted, False otherwise
        """
        try:
            query = """
            DELETE FROM dashboards
            WHERE id = %s AND user_id = %s
            """

            self.db.execute_query(query, (dashboard_id, user_id))
            return True

        except Exception as e:
            logger.error(f"Error deleting dashboard: {e}")
            return False

    def add_widget_to_dashboard(
        self,
        dashboard_id: str,
        widget_type: WidgetType,
        title: str,
        size: WidgetSize = WidgetSize.MEDIUM,
        refresh_interval: RefreshInterval = RefreshInterval.MINUTE_5,
        data_source: Dict[str, Any] = None,
        display_options: Dict[str, Any] = None,
        position: Dict[str, int] = None,
    ) -> Optional[Widget]:
        """
        Add widget to dashboard

        Args:
            dashboard_id: Dashboard ID
            widget_type: Type of widget
            title: Widget title
            size: Widget size
            refresh_interval: Refresh interval
            data_source: Data source configuration
            display_options: Display options
            position: Grid position

        Returns:
            Created widget or None
        """
        try:
            widget_id = str(uuid4())

            # Auto-position if not specified
            if position is None:
                position = self._find_next_widget_position(dashboard_id, size)

            query = """
            INSERT INTO widgets (
                id, dashboard_id, type, title, size,
                refresh_interval, data_source, display_options, position
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """

            result = self.db.fetch_one(
                query,
                (
                    widget_id,
                    dashboard_id,
                    widget_type,
                    title,
                    size,
                    refresh_interval,
                    json.dumps(data_source or {}),
                    json.dumps(display_options or {}),
                    json.dumps(position),
                ),
            )

            return Widget(
                id=result["id"],
                dashboard_id=result["dashboard_id"],
                type=result["type"],
                title=result["title"],
                size=result["size"],
                refresh_interval=result["refresh_interval"],
                data_source=result["data_source"],
                display_options=result["display_options"],
                position=result["position"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )

        except Exception as e:
            logger.error(f"Error adding widget: {e}")
            return None

    def get_dashboard_widgets(self, dashboard_id: str) -> List[Widget]:
        """
        Get all widgets for a dashboard

        Args:
            dashboard_id: Dashboard ID

        Returns:
            List of widgets
        """
        try:
            query = """
            SELECT * FROM widgets
            WHERE dashboard_id = %s
            ORDER BY position->>'y', position->>'x'
            """

            results = self.db.fetch_all(query, (dashboard_id,))

            return [
                Widget(
                    id=row["id"],
                    dashboard_id=row["dashboard_id"],
                    type=row["type"],
                    title=row["title"],
                    size=row["size"],
                    refresh_interval=row["refresh_interval"],
                    data_source=row["data_source"],
                    display_options=row["display_options"],
                    position=row["position"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error getting widgets: {e}")
            return []

    def update_widget(self, widget_id: str, dashboard_id: str, **updates) -> Optional[Widget]:
        """
        Update widget configuration

        Args:
            widget_id: Widget ID
            dashboard_id: Dashboard ID for verification
            **updates: Fields to update

        Returns:
            Updated widget or None
        """
        try:
            allowed_fields = ["title", "size", "refresh_interval", "data_source", "display_options", "position"]

            update_fields = []
            params = []

            for field, value in updates.items():
                if field in allowed_fields:
                    update_fields.append(f"{field} = %s")
                    if field in ["data_source", "display_options", "position"]:
                        params.append(json.dumps(value))
                    else:
                        params.append(value)

            if not update_fields:
                return None

            params.extend([widget_id, dashboard_id])

            query = """
            UPDATE widgets
            SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND dashboard_id = %s
            RETURNING *
            """

            result = self.db.fetch_one(query, params)

            if not result:
                return None

            return Widget(
                id=result["id"],
                dashboard_id=result["dashboard_id"],
                type=result["type"],
                title=result["title"],
                size=result["size"],
                refresh_interval=result["refresh_interval"],
                data_source=result["data_source"],
                display_options=result["display_options"],
                position=result["position"],
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )

        except Exception as e:
            logger.error(f"Error updating widget: {e}")
            return None

    def delete_widget(self, widget_id: str, dashboard_id: str) -> bool:
        """
        Delete widget from dashboard

        Args:
            widget_id: Widget ID
            dashboard_id: Dashboard ID for verification

        Returns:
            True if deleted, False otherwise
        """
        try:
            query = """
            DELETE FROM widgets
            WHERE id = %s AND dashboard_id = %s
            """

            self.db.execute_query(query, (widget_id, dashboard_id))
            return True

        except Exception as e:
            logger.error(f"Error deleting widget: {e}")
            return False

    def share_dashboard(
        self, dashboard_id: str, owner_id: str, shared_with_user_ids: List[str], permissions: List[str] = None
    ) -> bool:
        """
        Share dashboard with other users

        Args:
            dashboard_id: Dashboard ID
            owner_id: Owner user ID
            shared_with_user_ids: User IDs to share with
            permissions: Permissions to grant

        Returns:
            True if shared successfully
        """
        try:
            # Verify ownership
            query = "SELECT id FROM dashboards WHERE id = %s AND user_id = %s"
            if not self.db.fetch_one(query, (dashboard_id, owner_id)):
                return False

            # Remove existing shares
            self.db.execute_query("DELETE FROM dashboard_shares WHERE dashboard_id = %s", (dashboard_id,))

            # Add new shares
            if shared_with_user_ids:
                values = []
                for user_id in shared_with_user_ids:
                    values.append((str(uuid4()), dashboard_id, user_id, json.dumps(permissions or ["view"]), owner_id))

                query = """
                INSERT INTO dashboard_shares 
                (id, dashboard_id, shared_with_user_id, permissions, shared_by)
                VALUES (%s, %s, %s, %s, %s)
                """

                for value in values:
                    self.db.execute_query(query, value)

            return True

        except Exception as e:
            logger.error(f"Error sharing dashboard: {e}")
            return False

    def get_dashboard_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available dashboard templates

        Args:
            category: Optional category filter

        Returns:
            List of templates
        """
        try:
            if category:
                query = """
                SELECT * FROM dashboard_templates
                WHERE category = %s AND is_active = true
                ORDER BY name
                """
                params = (category,)
            else:
                query = """
                SELECT * FROM dashboard_templates
                WHERE is_active = true
                ORDER BY category, name
                """
                params = ()

            results = self.db.fetch_all(query, params)

            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "category": row["category"],
                    "preview_image": row.get("preview_image"),
                    "widgets": row["widgets"],
                    "tags": row["tags"],
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error getting templates: {e}")
            return []

    def create_dashboard_from_template(
        self, organization_id: str, user_id: str, template_id: str, name: str, **overrides
    ) -> Optional[DashboardConfig]:
        """
        Create dashboard from template

        Args:
            organization_id: Organization ID
            user_id: User ID
            template_id: Template ID
            name: Dashboard name
            **overrides: Override template settings

        Returns:
            Created dashboard or None
        """
        try:
            # Get template
            query = "SELECT * FROM dashboard_templates WHERE id = %s"
            template = self.db.fetch_one(query, (template_id,))

            if not template:
                return None

            # Create dashboard with template widgets
            dashboard = self.create_dashboard(
                organization_id=organization_id,
                user_id=user_id,
                name=name,
                description=overrides.get("description", template["description"]),
                layout=overrides.get("layout", DashboardLayout.GRID),
                theme=overrides.get("theme", DashboardTheme.LIGHT),
                is_default=overrides.get("is_default", False),
                is_public=overrides.get("is_public", False),
                tags=template["tags"],
                widgets=[WidgetConfig(**w) for w in template["widgets"]],
                settings=overrides.get("settings", {}),
            )

            return dashboard

        except Exception as e:
            logger.error(f"Error creating from template: {e}")
            return None

    def _find_next_widget_position(self, dashboard_id: str, widget_size: WidgetSize) -> Dict[str, int]:
        """
        Find next available position for widget

        Args:
            dashboard_id: Dashboard ID
            widget_size: Widget size

        Returns:
            Position dict with x, y coordinates
        """
        try:
            # Get existing widget positions
            query = """
            SELECT position, size FROM widgets
            WHERE dashboard_id = %s
            """

            results = self.db.fetch_all(query, (dashboard_id,))

            # Simple grid allocation (12 columns)
            occupied = set()
            for row in results:
                pos = row["position"]
                size = row["size"]

                # Calculate occupied cells
                width = self._get_widget_width(size)
                height = self._get_widget_height(size)

                for y in range(pos["y"], pos["y"] + height):
                    for x in range(pos["x"], pos["x"] + width):
                        occupied.add((x, y))

            # Find first available position
            widget_width = self._get_widget_width(widget_size)
            widget_height = self._get_widget_height(widget_size)

            for y in range(0, 100):  # Max 100 rows
                for x in range(0, 12 - widget_width + 1):
                    # Check if position is available
                    available = True
                    for dy in range(widget_height):
                        for dx in range(widget_width):
                            if (x + dx, y + dy) in occupied:
                                available = False
                                break
                        if not available:
                            break

                    if available:
                        return {"x": x, "y": y}

            # Fallback to end
            return {"x": 0, "y": 100}

        except Exception as e:
            logger.error(f"Error finding position: {e}")
            return {"x": 0, "y": 0}

    def _get_widget_width(self, size: WidgetSize) -> int:
        """Get widget width in grid units"""
        widths = {
            WidgetSize.SMALL: 1,
            WidgetSize.MEDIUM: 2,
            WidgetSize.LARGE: 2,
            WidgetSize.WIDE: 3,
            WidgetSize.TALL: 1,
            WidgetSize.EXTRA_LARGE: 3,
        }
        return widths.get(size, 2)

    def _get_widget_height(self, size: WidgetSize) -> int:
        """Get widget height in grid units"""
        heights = {
            WidgetSize.SMALL: 1,
            WidgetSize.MEDIUM: 1,
            WidgetSize.LARGE: 2,
            WidgetSize.WIDE: 1,
            WidgetSize.TALL: 2,
            WidgetSize.EXTRA_LARGE: 2,
        }
        return heights.get(size, 1)
