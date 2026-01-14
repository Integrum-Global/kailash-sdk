"""Audit middleware for tracking requests."""

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for auditing requests."""

    def __init__(self, app, audit_logger):
        super().__init__(app)
        self.audit_logger = audit_logger

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log audit events."""
        # Log request
        await self.audit_logger.log_event(
            event_type="http_request",
            resource=str(request.url),
            action=request.method,
            details={
                "path": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else None,
            },
        )

        # Process request
        response = await call_next(request)
        return response
