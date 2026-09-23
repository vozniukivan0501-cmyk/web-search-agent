"""Agent interface wrapping the LangGraph pipeline."""
import logging
from typing import AsyncGenerator, Dict, Any
import json

from app.graph import build_agent_graph, AgentState

logger = logging.getLogger(__name__)

agent_graph = build_agent_graph()


async def run_agent(query: str, api_key: str) -> Dict[str, Any]:
    """Run the agent and return the final result."""
    initial_state: AgentState = {
        "original_query": query,
        "sub_queries": [],
        "search_results": [],
        "reranked_results": [],
        "page_contents": [],
        "answer": "",
        "reasoning_steps": [],
        "step_count": 0,
        "is_complete": False,
        "api_key": api_key,
        "status": "Starting..."
    }
    
    final_state = await agent_graph.ainvoke(initial_state)
    
    return {
        "answer": final_state.get("answer", "No answer generated."),
        "reasoning_steps": final_state.get("reasoning_steps", []),
        "sources": [
            {"title": r.get("title", ""), "url": r.get("url", "")}
            for r in final_state.get("reranked_results", [])
        ],
        "steps_taken": final_state.get("step_count", 0)
    }


async def run_agent_stream(query: str, api_key: str) -> AsyncGenerator[str, None]:
    """Run the agent with streaming status updates via SSE."""
    initial_state: AgentState = {
        "original_query": query,
        "sub_queries": [],
        "search_results": [],
        "reranked_results": [],
        "page_contents": [],
        "answer": "",
        "reasoning_steps": [],
        "step_count": 0,
        "is_complete": False,
        "api_key": api_key,
        "status": "Starting..."
    }
    
    current_state = dict(initial_state)
    last_status = "Analyzing question and planning search strategy..."
    yield json.dumps({"type": "status", "node": "init", "message": last_status})
    
    async for event in agent_graph.astream(initial_state):
        # event is a dict with node name as key and state update as value
        for node_name, state_update in event.items():
            current_state.update(state_update)
            status = state_update.get("status", "")
            if status and status != last_status:
                last_status = status
                yield json.dumps({"type": "status", "node": node_name, "message": status})
            
            # If this is the final state with an answer
            if "answer" in state_update and state_update.get("is_complete", False):
                yield json.dumps({
                    "type": "result",
                    "answer": state_update["answer"],
                    "reasoning_steps": state_update.get("reasoning_steps", []),
                })
    
    # Send final result after stream completes
    final_result = {
        "answer": current_state.get("answer", "No answer generated."),
        "reasoning_steps": current_state.get("reasoning_steps", []),
        "sources": [
            {"title": r.get("title", ""), "url": r.get("url", "")}
            for r in current_state.get("reranked_results", [])
        ],
        "steps_taken": current_state.get("step_count", 0)
    }
    yield json.dumps({"type": "final", **final_result})
