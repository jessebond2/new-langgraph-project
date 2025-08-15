import pytest

from agent import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    inputs = {
        "query": "test search",
        "num_requests": 5,  # Use fewer requests for testing
        "limit": 50,
        "locale": "en-US"
    }
    res = await graph.ainvoke(inputs)
    assert res is not None
    assert "summary" in res
    assert "request_params" in res
    assert "basic_stats" in res
    assert "detailed_lifecycle_stats" in res
    assert "individual_request_details" in res
