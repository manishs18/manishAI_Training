# Extension Tasks

# Extension A-Rate-Limiting Middleware (Medium)

# Add a sliding-window rate limiter (e.g., 100 req/min per client IP) using an in-memory dictionary.

# Return 429 Too Many Requests with a Retry-After header when the limit is exceeded

# Track limit hits as a Prometheus Counter

# Extension B - Correlation ID Propagation (Medium)

# The current middleware assigns a correlation ID but does NOT forward it to outbound httpx calls.

# Add a contextvars.Contextvar to store the correlation ID per request

# Write a custom httpx.Auth or event hook that injects x-Correlation-Id into every outbound call

# Verify it appears in mock downstream requests

# Variables

# Extension つ

# Dromathaus Grafana Dashhnard/Ardvanced)

# 29C

# Terminal



# app.py
import time
import uuid
from collections import defaultdict, deque
from contextvars import ContextVar

import httpx
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from prometheus_client import Counter, Histogram, start_http_server

# -----------------------------
# Prometheus Metrics
# -----------------------------
rate_limit_hits = Counter(
    "rate_limit_hits_total", "Number of rate limit violations"
)
http_requests_total = Counter(
    "http_requests_total", "Total number of HTTP requests", ["path", "method", "status"]
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", ["path"]
)

# -----------------------------
# Rate Limiter (Sliding Window)
# -----------------------------
WINDOW_SECONDS = 60
MAX_REQUESTS = 100
request_log = defaultdict(deque)

# -----------------------------
# Correlation ID ContextVar
# -----------------------------
correlation_id_ctx = ContextVar("correlation_id", default=None)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()


# -----------------------------
# Rate Limit Middleware
# -----------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()

        requests = request_log[client_ip]

        # Remove old requests
        while requests and requests[0] < now - WINDOW_SECONDS:
            requests.popleft()

        if len(requests) >= MAX_REQUESTS:
            retry_after = int(WINDOW_SECONDS - (now - requests[0]))
            rate_limit_hits.inc()
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        requests.append(now)
        return await call_next(request)


# -----------------------------
# Correlation ID Middleware
# -----------------------------
class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
        correlation_id_ctx.set(correlation_id)
        response = await call_next(request)
        response.headers["x-correlation-id"] = correlation_id
        return response


# -----------------------------
# Metrics Middleware
# -----------------------------
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        http_requests_total.labels(
            path=request.url.path,
            method=request.method,
            status=str(response.status_code),
        ).inc()

        http_request_duration_seconds.labels(
            path=request.url.path
        ).observe(duration)

        return response


# -----------------------------
# Register Middleware
# -----------------------------
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(MetricsMiddleware)

# -----------------------------
# HTTPX Client with Correlation ID Propagation
# -----------------------------
async def add_correlation_id(request: httpx.Request):
    correlation_id = correlation_id_ctx.get()
    if correlation_id:
        request.headers["x-correlation-id"] = correlation_id


http_client = httpx.AsyncClient(event_hooks={"request": [add_correlation_id]})


# -----------------------------
# Example Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Hello, world!"}


@app.get("/proxy")
async def proxy():
    # Example outbound request
    resp = await http_client.get("https://httpbin.org/get")
    return {"proxied_response": resp.json()}


@app.get("/echo")
async def echo(request: Request):
    # Return correlation ID received by this endpoint
    return {"correlation_id": request.headers.get("x-correlation-id")}


# -----------------------------
# Start Prometheus Metrics Server
# -----------------------------
start_http_server(8001)  # Prometheus scrapes metrics here


# -----------------------------
# Run with: uvicorn app:app --reload
# -----------------------------