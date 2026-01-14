"""Tenant isolation middleware."""

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware for tenant isolation."""

    def __init__(self, app, tenant_manager):
        super().__init__(app)
        self.tenant_manager = tenant_manager

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Extract tenant context and process request."""
        # Extract tenant from header or subdomain
        tenant_id = request.headers.get("X-Tenant-ID")

        if not tenant_id:
            # Try to extract from subdomain
            host = request.headers.get("host", "")
            if "." in host:
                tenant_id = host.split(".")[0]

        # Set tenant context
        request.state.tenant_id = tenant_id or "default"

        # Process request
        response = await call_next(request)
        return response
