#!/usr/bin/env python3
"""Basic MCP Server - Simple implementation with core tools."""

import asyncio
import logging
from typing import Any, Dict, Optional

from kailash.mcp_server import MCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BasicMCPServer(MCPServer):
    """Basic MCP server with simple tools."""

    def __init__(
        self, name: str = "basic-mcp-server", config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name)
        self.config = config or {}
        self._register_tools()

    def _register_tools(self):
        """Register basic tools."""

        # Math tools
        @self.tool()
        async def add(a: float, b: float) -> Dict[str, Any]:
            """Add two numbers together."""
            return {"result": a + b}

        @self.tool()
        async def subtract(a: float, b: float) -> Dict[str, Any]:
            """Subtract b from a."""
            return {"result": a - b}

        @self.tool()
        async def multiply(a: float, b: float) -> Dict[str, Any]:
            """Multiply two numbers."""
            return {"result": a * b}

        @self.tool()
        async def divide(a: float, b: float) -> Dict[str, Any]:
            """Divide a by b."""
            if b == 0:
                raise ValueError("Division by zero")
            return {"result": a / b}

        # String tools
        @self.tool()
        async def uppercase(text: str) -> Dict[str, Any]:
            """Convert text to uppercase."""
            return {"result": text.upper()}

        @self.tool()
        async def lowercase(text: str) -> Dict[str, Any]:
            """Convert text to lowercase."""
            return {"result": text.lower()}

        @self.tool()
        async def reverse(text: str) -> Dict[str, Any]:
            """Reverse a string."""
            return {"result": text[::-1]}

        @self.tool()
        async def count_words(text: str) -> Dict[str, Any]:
            """Count words in text."""
            return {"result": len(text.split())}

        # Data tools
        @self.tool()
        async def get_timestamp() -> Dict[str, Any]:
            """Get current timestamp."""
            from datetime import datetime

            return {"result": datetime.now().isoformat()}

        @self.tool()
        async def echo(message: str) -> Dict[str, Any]:
            """Echo back the message."""
            return {"result": message}


def create_basic_server() -> MCPServer:
    """Create a basic MCP server with simple tools."""
    server = MCPServer("basic-mcp-server")

    # Register math tools
    @server.tool()
    async def add(a: float, b: float) -> Dict[str, Any]:
        """Add two numbers together."""
        return {"result": a + b}

    @server.tool()
    async def subtract(a: float, b: float) -> Dict[str, Any]:
        """Subtract b from a."""
        return {"result": a - b}

    @server.tool()
    async def multiply(a: float, b: float) -> Dict[str, Any]:
        """Multiply two numbers."""
        return {"result": a * b}

    @server.tool()
    async def divide(a: float, b: float) -> Dict[str, Any]:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Division by zero")
        return {"result": a / b}

    # Register string tools
    @server.tool()
    async def uppercase(text: str) -> Dict[str, Any]:
        """Convert text to uppercase."""
        return {"result": text.upper()}

    @server.tool()
    async def lowercase(text: str) -> Dict[str, Any]:
        """Convert text to lowercase."""
        return {"result": text.lower()}

    @server.tool()
    async def reverse(text: str) -> Dict[str, Any]:
        """Reverse a string."""
        return {"result": text[::-1]}

    # Register utility tools
    @server.tool()
    async def echo(message: str) -> Dict[str, Any]:
        """Echo back the provided message."""
        return {"message": message, "echo": True}

    @server.tool()
    async def get_info() -> Dict[str, Any]:
        """Get server information."""
        return {
            "name": "Basic MCP Server",
            "version": "1.0.0",
            "tools_count": len(server._tool_registry),
            "status": "running",
        }

    # Register a resource
    @server.resource("config://server")
    async def server_config() -> Dict[str, Any]:
        """Provide server configuration as a resource."""
        return {
            "name": "basic-mcp-server",
            "capabilities": {
                "math": ["add", "subtract", "multiply", "divide"],
                "string": ["uppercase", "lowercase", "reverse"],
                "utility": ["echo", "get_info"],
            },
            "limits": {"max_number": 1e9, "max_string_length": 10000},
        }

    logger.info(f"Created basic server with {len(server._tool_registry)} tools")
    return server


def main():
    """Run the basic MCP server."""
    server = create_basic_server()

    try:
        logger.info("Starting basic MCP server...")
        # Use the run method which is available on MCPServer
        server.run()

    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
