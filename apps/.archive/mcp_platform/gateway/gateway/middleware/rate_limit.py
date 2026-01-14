"""Rate limiting middleware."""

import time
from collections import defaultdict
from typing import Callable, Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app, rate_limit: int = 100, window: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.window = window
        self.requests: Dict[str, list] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limits and process request."""
        client_id = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old requests
        self.requests[client_id] = [
            req_time
            for req_time in self.requests[client_id]
            if req_time > now - self.window
        ]

        # Check rate limit
        if len(self.requests[client_id]) >= self.rate_limit:
            return Response(
                content="Rate limit exceeded",
                status_code=429,
            )

        # Track request
        self.requests[client_id].append(now)

        # Process request
        response = await call_next(request)
        return response
