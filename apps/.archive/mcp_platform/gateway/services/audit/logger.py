"""Minimal audit logger for MCP Platform."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Simple audit logger for tracking system events."""

    def __init__(self):
        self.events = []

    async def log_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "details": details or {},
        }
        self.events.append(event)
        logger.info(f"Audit event: {event_type}", extra=event)

    async def get_events(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """Get audit events."""
        events = self.events
        if user_id:
            events = [e for e in events if e.get("user_id") == user_id]
        return events[-limit:]
