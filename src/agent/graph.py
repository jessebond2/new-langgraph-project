"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict

import requests
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    jwt: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """

    jwt = runtime.context.get("jwt")
    if not jwt:
        raise ValueError("JWT is required")
    
    # Call search api
    search_api_url = "https://api.prod.headspace.com/recall/v1/searches?query=langgraph&locale=en-US&limit=100"
    headers = {
        "Authorization": f"Bearer {jwt}"
    }
    
    try:
        response = requests.get(search_api_url, headers=headers)
        response.raise_for_status()
        search_results = response.json()
        
        return {
            "changeme": f"Search API call completed successfully. Found {len(search_results.get('results', []))} results from {search_api_url}"
        }
    except requests.RequestException as e:
        return {
            "changeme": f"Error calling search API: {str(e)}"
        }


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="New Graph")
)
