import structlog
import uuid
from flask import g, request

def configure_logger():
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(10),
    )

def add_correlation_id():
    """Middleware-like helper to inject correlation_id into the log context."""
    if not hasattr(g, "correlation_id"):
        # Check header first, or generate a new one
        g.correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    return {"correlation_id": g.correlation_id}