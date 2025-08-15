"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import aiohttp
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime


@dataclass
class RequestLifecycle:
    """Detailed tracking of HTTP request lifecycle phases."""
    request_id: int
    start_time: float = 0.0
    dns_resolution_start: Optional[float] = None
    dns_resolution_end: Optional[float] = None
    connection_create_start: Optional[float] = None
    connection_create_end: Optional[float] = None
    request_start: Optional[float] = None
    request_end: Optional[float] = None
    response_start: Optional[float] = None
    response_end: Optional[float] = None
    connection_reused: bool = False
    response_headers: Dict[str, str] = field(default_factory=dict)
    request_headers: Dict[str, str] = field(default_factory=dict)
    status_code: Optional[int] = None
    response_size: int = 0
    url: str = ""
    method: str = "GET"
    error: Optional[str] = None


class RequestTracer:
    """Tracks detailed timing information for HTTP requests."""
    
    def __init__(self, lifecycle: RequestLifecycle):
        self.lifecycle = lifecycle
        
    async def on_dns_resolvehost_start(self, session, trace_config_ctx, params):
        self.lifecycle.dns_resolution_start = time.time()
        
    async def on_dns_resolvehost_end(self, session, trace_config_ctx, params):
        self.lifecycle.dns_resolution_end = time.time()
        
    async def on_connection_create_start(self, session, trace_config_ctx, params):
        self.lifecycle.connection_create_start = time.time()
        
    async def on_connection_create_end(self, session, trace_config_ctx, params):
        self.lifecycle.connection_create_end = time.time()
        
    async def on_connection_reuseconn(self, session, trace_config_ctx, params):
        self.lifecycle.connection_reused = True
        
    async def on_request_start(self, session, trace_config_ctx, params):
        self.lifecycle.request_start = time.time()
        self.lifecycle.method = str(params.method)
        self.lifecycle.url = str(params.url)
        self.lifecycle.request_headers = dict(params.headers)
        
    async def on_request_end(self, session, trace_config_ctx, params):
        self.lifecycle.request_end = time.time()
        
    async def on_response_chunk_received(self, session, trace_config_ctx, params):
        if self.lifecycle.response_start is None:
            self.lifecycle.response_start = time.time()
        self.lifecycle.response_size += len(params.chunk)


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """



@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    query: str = "langgraph"
    num_requests: int = 25
    limit: int = 100
    locale: str = "en-US"


async def make_single_request_with_tracing(session: aiohttp.ClientSession, url: str, headers: Dict[str, str], request_id: int) -> Dict[str, Any]:
    """Make a single API request with detailed lifecycle tracking."""
    
    # Initialize lifecycle tracking
    lifecycle = RequestLifecycle(request_id=request_id)
    lifecycle.start_time = time.time()
    
    # Set up tracing
    tracer = RequestTracer(lifecycle)
    trace_config = aiohttp.TraceConfig()
    trace_config.on_dns_resolvehost_start.append(tracer.on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(tracer.on_dns_resolvehost_end)
    trace_config.on_connection_create_start.append(tracer.on_connection_create_start)
    trace_config.on_connection_create_end.append(tracer.on_connection_create_end)
    trace_config.on_connection_reuseconn.append(tracer.on_connection_reuseconn)
    trace_config.on_request_start.append(tracer.on_request_start)
    trace_config.on_request_end.append(tracer.on_request_end)
    trace_config.on_response_chunk_received.append(tracer.on_response_chunk_received)
    
    try:
        # Create a session with tracing for this specific request
        async with aiohttp.ClientSession(trace_configs=[trace_config]) as traced_session:
            async with traced_session.get(url, headers=headers) as response:
                lifecycle.response_end = time.time()
                lifecycle.status_code = response.status
                lifecycle.response_headers = dict(response.headers)
                
                response.raise_for_status()
                data = await response.json()
                
                # Calculate detailed timings
                total_duration = lifecycle.response_end - lifecycle.start_time
                
                # Calculate phase durations
                dns_duration = None
                if lifecycle.dns_resolution_start and lifecycle.dns_resolution_end:
                    dns_duration = lifecycle.dns_resolution_end - lifecycle.dns_resolution_start
                
                connection_duration = None
                if lifecycle.connection_create_start and lifecycle.connection_create_end:
                    connection_duration = lifecycle.connection_create_end - lifecycle.connection_create_start
                
                request_duration = None
                if lifecycle.request_start and lifecycle.request_end:
                    request_duration = lifecycle.request_end - lifecycle.request_start
                
                response_duration = None
                if lifecycle.response_start and lifecycle.response_end:
                    response_duration = lifecycle.response_end - lifecycle.response_start
                
                time_to_first_byte = None
                if lifecycle.response_start:
                    time_to_first_byte = lifecycle.response_start - lifecycle.start_time
                
                return {
                    "request_id": request_id,
                    "success": True,
                    "total_duration": total_duration,
                    "status_code": response.status,
                    "result_count": len(data.get('results', [])),
                    "error": None,
                    "lifecycle_details": {
                        "dns_resolution_duration": dns_duration,
                        "connection_duration": connection_duration,
                        "request_duration": request_duration,
                        "response_duration": response_duration,
                        "time_to_first_byte": time_to_first_byte,
                        "connection_reused": lifecycle.connection_reused,
                        "response_size_bytes": lifecycle.response_size,
                        "method": lifecycle.method,
                        "url": lifecycle.url
                    },
                    "headers": {
                        "request_headers": lifecycle.request_headers,
                        "response_headers": lifecycle.response_headers
                    },
                    "timestamps": {
                        "start_time": lifecycle.start_time,
                        "dns_start": lifecycle.dns_resolution_start,
                        "dns_end": lifecycle.dns_resolution_end,
                        "connection_start": lifecycle.connection_create_start,
                        "connection_end": lifecycle.connection_create_end,
                        "request_start": lifecycle.request_start,
                        "request_end": lifecycle.request_end,
                        "response_start": lifecycle.response_start,
                        "response_end": lifecycle.response_end
                    }
                }
                
    except Exception as e:
        total_duration = time.time() - lifecycle.start_time
        lifecycle.error = str(e)
        
        return {
            "request_id": request_id,
            "success": False,
            "total_duration": total_duration,
            "status_code": lifecycle.status_code,
            "result_count": 0,
            "error": str(e),
            "lifecycle_details": {
                "dns_resolution_duration": None,
                "connection_duration": None,
                "request_duration": None,
                "response_duration": None,
                "time_to_first_byte": None,
                "connection_reused": lifecycle.connection_reused,
                "response_size_bytes": lifecycle.response_size,
                "method": lifecycle.method or "GET",
                "url": lifecycle.url or url
            },
            "headers": {
                "request_headers": lifecycle.request_headers,
                "response_headers": lifecycle.response_headers
            },
            "timestamps": {
                "start_time": lifecycle.start_time,
                "dns_start": lifecycle.dns_resolution_start,
                "dns_end": lifecycle.dns_resolution_end,
                "connection_start": lifecycle.connection_create_start,
                "connection_end": lifecycle.connection_create_end,
                "request_start": lifecycle.request_start,
                "request_end": lifecycle.request_end,
                "response_start": lifecycle.response_start,
                "response_end": lifecycle.response_end
            }
        }


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Makes concurrent API requests using data from the initial request and tracks individual and average latencies.
    """

    jwt = os.getenv("JWT_TOKEN")
    if not jwt:
        raise ValueError("JWT_TOKEN environment variable is required but not set")
    
    # Build search URL using data from initial request
    search_api_url = f"https://api.prod.headspace.com/recall/v1/searches?query={state.query}&locale={state.locale}&limit={state.limit}"
    headers = {
        "Authorization": f"Bearer {jwt}"
    }
    
    # Record overall start time
    overall_start_time = time.time()
    
    # Make concurrent requests using num_requests from initial request with detailed tracking
    tasks = [
        make_single_request_with_tracing(None, search_api_url, headers, i+1) 
        for i in range(state.num_requests)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Calculate overall duration
    overall_duration = time.time() - overall_start_time
    
    # Extract latencies and calculate statistics from detailed results
    latencies = [result["total_duration"] for result in results]
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    # Calculate basic statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    success_rate = len(successful_requests) / len(results) * 100
    
    # Calculate detailed lifecycle statistics for successful requests
    def calculate_phase_stats(phase_name: str):
        phase_durations = []
        for r in successful_requests:
            phase_duration = r.get("lifecycle_details", {}).get(phase_name)
            if phase_duration is not None:
                phase_durations.append(phase_duration)
        
        if phase_durations:
            return {
                "avg": sum(phase_durations) / len(phase_durations),
                "min": min(phase_durations),
                "max": max(phase_durations),
                "count": len(phase_durations)
            }
        return None
    
    # Detailed lifecycle statistics
    dns_stats = calculate_phase_stats("dns_resolution_duration")
    connection_stats = calculate_phase_stats("connection_duration")
    request_stats = calculate_phase_stats("request_duration")
    response_stats = calculate_phase_stats("response_duration")
    ttfb_stats = calculate_phase_stats("time_to_first_byte")
    
    # Connection reuse statistics
    connection_reused_count = sum(1 for r in successful_requests 
                                 if r.get("lifecycle_details", {}).get("connection_reused", False))
    
    # Response size statistics
    response_sizes = [r.get("lifecycle_details", {}).get("response_size_bytes", 0) 
                     for r in successful_requests]
    total_response_size = sum(response_sizes)
    avg_response_size = total_response_size / len(response_sizes) if response_sizes else 0
    
    # Create detailed summary
    summary = (
        f"Completed {state.num_requests} concurrent API requests for query '{state.query}' in {overall_duration:.3f}s total. "
        f"Success rate: {success_rate:.1f}% ({len(successful_requests)}/{state.num_requests}). "
        f"Latencies - Avg: {avg_latency:.3f}s, Min: {min_latency:.3f}s, Max: {max_latency:.3f}s. "
    )
    
    if successful_requests:
        total_results = sum(r["result_count"] for r in successful_requests)
        summary += f"Total results found: {total_results}. "
    
    if failed_requests:
        summary += f"Errors: {len(failed_requests)} requests failed."
    
    return {
        "summary": summary,
        "request_params": {
            "query": state.query,
            "num_requests": state.num_requests,
            "limit": state.limit,
            "locale": state.locale,
            "search_url": search_api_url
        },
        "basic_stats": {
            "individual_latencies": latencies,
            "average_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "overall_duration": overall_duration,
            "success_rate": success_rate,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests)
        },
        "detailed_lifecycle_stats": {
            "dns_resolution": dns_stats,
            "connection_establishment": connection_stats,
            "request_transmission": request_stats,
            "response_reception": response_stats,
            "time_to_first_byte": ttfb_stats,
            "connection_reuse": {
                "reused_connections": connection_reused_count,
                "new_connections": len(successful_requests) - connection_reused_count,
                "reuse_rate": (connection_reused_count / len(successful_requests) * 100) if successful_requests else 0
            },
            "response_sizes": {
                "total_bytes": total_response_size,
                "average_bytes": avg_response_size,
                "individual_sizes": response_sizes
            }
        },
        "individual_request_details": results
    }


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="New Graph")
)
