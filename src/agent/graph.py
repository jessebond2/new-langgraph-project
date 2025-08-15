"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

import aiohttp
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime


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
    jwt: str = ""


async def make_single_request(session: aiohttp.ClientSession, url: str, headers: Dict[str, str], request_id: int) -> Dict[str, Any]:
    """Make a single API request and track its latency."""
    start_time = time.time()
    
    try:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            duration = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": True,
                "duration": duration,
                "status_code": response.status,
                "result_count": len(data.get('results', [])),
                "error": None
            }
    except Exception as e:
        duration = time.time() - start_time
        
        return {
            "request_id": request_id,
            "success": False,
            "duration": duration,
            "status_code": None,
            "result_count": 0,
            "error": str(e)
        }


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Makes concurrent API requests using data from the initial request and tracks individual and average latencies.
    """

    jwt = state.jwt
    if not jwt:
        raise ValueError("JWT is required")
    
    # Build search URL using data from initial request
    search_api_url = f"https://api.prod.headspace.com/recall/v1/searches?query={state.query}&locale={state.locale}&limit={state.limit}"
    headers = {
        "Authorization": f"Bearer {jwt}"
    }
    
    # Record overall start time
    overall_start_time = time.time()
    
    # Make concurrent requests using num_requests from initial request
    async with aiohttp.ClientSession() as session:
        tasks = [
            make_single_request(session, search_api_url, headers, i+1) 
            for i in range(state.num_requests)
        ]
        
        results = await asyncio.gather(*tasks)
    
    # Calculate overall duration
    overall_duration = time.time() - overall_start_time
    
    # Extract latencies and calculate statistics
    latencies = [result["duration"] for result in results]
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    success_rate = len(successful_requests) / len(results) * 100
    
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
        "latency_stats": {
            "individual_latencies": latencies,
            "average_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "overall_duration": overall_duration,
            "success_rate": success_rate,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests)
        }
    }


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="New Graph")
)
