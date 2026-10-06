import asyncio
import json
import logging
from typing import TypedDict, Annotated, Literal, Any, List

from langgraph.graph import StateGraph, START, END
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, Field, ValidationError

from app.search_tools import search_duckduckgo, fetch_page_content
from app.cross_encoder import rerank_results
from app.config import settings

logger = logging.getLogger(__name__)


class PlanResponse(BaseModel):
    sub_queries: List[str] = Field(default_factory=list)
    reasoning: str = "Direct search"


class EvaluateResponse(BaseModel):
    is_complete: bool = True
    missing: str = ""


def merge_lists(a: list, b: list) -> list:
    return a + b


def replace_value(a: Any, b: Any) -> Any:
    return b


class AgentState(TypedDict):
    original_query: Annotated[str, replace_value]
    sub_queries: Annotated[list[str], replace_value]
    search_results: Annotated[list[dict], merge_lists]
    reranked_results: Annotated[list[dict], replace_value]
    page_contents: Annotated[list[dict], merge_lists]
    answer: Annotated[str, replace_value]
    reasoning_steps: Annotated[list[str], merge_lists]
    step_count: Annotated[int, replace_value]
    is_complete: Annotated[bool, replace_value]
    api_key: Annotated[str, replace_value]
    status: Annotated[str, replace_value]


def get_llm(api_key: str) -> Gemini:
    return Gemini(
        api_key=api_key,
        model=settings.model_name,
    )


async def llm_generate(api_key: str, system_prompt: str, user_prompt: str, max_retries: int = 4) -> str:
    llm = get_llm(api_key)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]
    for attempt in range(max_retries):
        try:
            response = await llm.achat(messages)
            return response.message.content
        except Exception as e:
            err = str(e).lower()
            # Free tier hits 5 req/min quickly, so back off slightly
            if any(k in err for k in ("429", "quota", "resourceexhausted", "rate")) and attempt < max_retries - 1:
                delay = (attempt + 1) * 3
                logger.warning("Hit Gemini rate limit, sleeping %ss before retry (%d/%d)...", delay, attempt + 1, max_retries)
                await asyncio.sleep(delay)
            else:
                raise


async def plan_node(state: AgentState) -> dict:
    step = state.get("step_count", 0) + 1
    logger.info("Planning search step %d for query: %s", step, state["original_query"])
    
    prev_steps = state.get("reasoning_steps", [])
    context = ""
    if prev_steps:
        context = "\n\nPrevious steps:\n" + "\n".join(f"- {s}" for s in prev_steps)
        if state.get("answer"):
            context += f"\n\nIncomplete answer so far:\n{state['answer']}"
    
    system_prompt = (
        "You are a search planning agent. Break down the user question into 1-3 targeted web search queries.\n"
        "Return ONLY valid JSON without markdown wrapping.\n"
        'Format: {"sub_queries": ["query1", "query2"], "reasoning": "brief explanation"}'
    )
    user_prompt = f"User question: {state['original_query']}{context}"
    
    result = await llm_generate(state["api_key"], system_prompt, user_prompt)
    
    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
            
        parsed = PlanResponse.model_validate_json(clean)
        sub_queries = parsed.sub_queries or [state["original_query"]]
        reasoning = parsed.reasoning
    except (ValidationError, ValueError) as err:
        logger.warning("Failed to validate plan output (%s), using fallback", err)
        sub_queries = [state["original_query"]]
        reasoning = "Fallback to original query"
    
    return {
        "sub_queries": sub_queries[:3],
        "reasoning_steps": [f"[Step {step} - Plan] {reasoning}"],
        "step_count": step,
        "status": f"Planning search strategy (step {step})..."
    }


async def search_node(state: AgentState) -> dict:
    queries = state["sub_queries"]
    logger.info("Running DDG search for %d queries", len(queries))
    
    tasks = [search_duckduckgo(q, max_results=settings.max_search_results) for q in queries]
    results_lists = await asyncio.gather(*tasks)
    
    seen = set()
    deduped = []
    for items in results_lists:
        for item in items:
            url = item.get("url")
            if url and url not in seen:
                seen.add(url)
                deduped.append(item)
    
    # If sub-queries returned 0 results, search DuckDuckGo directly with original query
    if not deduped:
        logger.info("Sub-queries returned 0 results; searching DuckDuckGo with original query: %s", state["original_query"])
        direct_results = await search_duckduckgo(state["original_query"], max_results=settings.max_search_results)
        for item in direct_results:
            url = item.get("url")
            if url and url not in seen:
                seen.add(url)
                deduped.append(item)

    status_msg = f"Found {len(deduped)} search results, reranking..." if deduped else "No search results found."
    return {
        "search_results": deduped,
        "status": status_msg
    }


async def rerank_node(state: AgentState) -> dict:
    items = state.get("search_results", [])
    if not items:
        logger.info("No search results to rerank")
        return {
            "reranked_results": [],
            "reasoning_steps": ["[Rerank] No search results to rerank"],
            "status": "No search results found to fetch."
        }
    
    logger.info("Reranking %d results with cross-encoder", len(items))
    ranked = rerank_results(
        query=state["original_query"],
        results=items,
        model_name=settings.cross_encoder_model,
        top_k=settings.top_k_reranked
    )
    
    # Если реранкер нашел 0 валидных вкладок, опираемся на первые 3 вкладки из поиска DuckDuckGo
    reasoning_steps = []
    if not ranked and items:
        logger.info("Reranker found 0 valid tabs, falling back to top 3 search tabs.")
        ranked = items[:3]
        reasoning_steps.append(f"[Rerank] Found 0 valid tabs; relying on top {len(ranked)} DuckDuckGo tabs.")
    elif ranked:
        reasoning_steps.append(f"[Rerank] Kept top {len(ranked)} of {len(items)} results")
    else:
        reasoning_steps.append("[Rerank] 0 valid tabs found.")
    
    return {
        "reranked_results": ranked,
        "reasoning_steps": reasoning_steps,
        "status": f"Fetching content from top {len(ranked)} pages..."
    }


async def fetch_node(state: AgentState) -> dict:
    targets = state.get("reranked_results", [])
    # Если в reranked_results пусто, но есть результаты поиска, берем первые 3 вкладки
    if not targets:
        items = state.get("search_results", [])
        if items:
            targets = items[:3]
            
    if not targets:
        logger.info("No URLs to fetch")
        return {
            "page_contents": [],
            "status": "Synthesizing answer..."
        }
    
    logger.info("Fetching raw content for %d URLs", len(targets))
    tasks = [fetch_page_content(item["url"]) for item in targets]
    pages = await asyncio.gather(*tasks)
    
    page_contents = []
    for item, content in zip(targets, pages):
        text = content
        # Если страница вернула ошибку загрузки, используем сниппет вкладки
        if not text or text.startswith("[Error fetching page"):
            snippet = item.get("snippet", "")
            text = f"{snippet}\n{text}" if snippet else text
        page_contents.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": text
        })
    
    return {
        "page_contents": page_contents,
        "status": "Synthesizing answer from gathered information..."
    }


async def synthesize_node(state: AgentState) -> dict:
    logger.info("Synthesizing answer from sources")
    pages = state.get("page_contents", [])
    
    if pages:
        sources = []
        for idx, page in enumerate(pages, start=1):
            sources.append(
                f"=== Source {idx}: {page['title']} ===\n"
                f"URL: {page['url']}\n"
                f"{page['content']}\n"
            )
        system_prompt = (
            "You are an AI assistant answering questions based on web search results.\n"
            "Provide a clear, well-structured answer citing your sources.\n"
            "Answer in the same language as the question."
        )
        user_prompt = (
            f"Question: {state['original_query']}\n\n"
            f"Sources:\n{''.join(sources)}\n\n"
            "Please provide a comprehensive answer with source links."
        )
    else:
        system_prompt = (
            "You are an AI assistant answering questions.\n"
            "Provide a clear, well-structured answer based on your knowledge.\n"
            "Answer in the same language as the question."
        )
        user_prompt = (
            f"Question: {state['original_query']}\n\n"
            "No live web sources could be retrieved. Please provide a helpful, accurate, and comprehensive "
            "answer based on your knowledge, and briefly mention that web search results were unavailable."
        )

    answer = await llm_generate(state["api_key"], system_prompt, user_prompt)
    return {
        "answer": answer,
        "status": "Evaluating answer completeness..."
    }


async def evaluate_node(state: AgentState) -> dict:
    step = state.get("step_count", 1)
    logger.info("Evaluating answer quality (step %d/%d)", step, settings.max_reasoning_steps)
    
    # Если нет источников или мы уже обработали вкладки, не зацикливаемся
    if not state.get("page_contents") or step >= 2:
        return {
            "is_complete": True,
            "reasoning_steps": [f"[Evaluate] Step {step} complete. Finalizing answer."],
            "status": "Answer complete."
        }
    
    # Cap loop iterations
    if step >= settings.max_reasoning_steps:
        return {
            "is_complete": True,
            "reasoning_steps": [f"[Evaluate] Hit max steps ({settings.max_reasoning_steps}), wrapping up"],
            "status": "Answer complete (max steps reached)."
        }
    
    system_prompt = (
        "You evaluate whether the provided answer thoroughly addresses the original question.\n"
        "Return ONLY JSON: {\"is_complete\": true/false, \"missing\": \"what is missing if incomplete\"}"
    )
    user_prompt = f"Question: {state['original_query']}\n\nCurrent answer:\n{state['answer']}"

    result = await llm_generate(state["api_key"], system_prompt, user_prompt)
    
    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
            
        parsed = EvaluateResponse.model_validate_json(clean)
        is_complete = parsed.is_complete
        missing = parsed.missing
    except (ValidationError, ValueError):
        # If evaluation output format breaks, assume done to prevent infinite loops
        is_complete = True
        missing = ""
    
    note = f"[Evaluate Step {step}] " + ("Answer is complete." if is_complete else f"Needs refinement: {missing}")
    
    return {
        "is_complete": is_complete,
        "reasoning_steps": [note],
        "status": "Answer complete!" if is_complete else f"Answer incomplete, refining (step {step})..."
    }


def should_continue(state: AgentState) -> Literal["plan", "end"]:
    return "end" if state.get("is_complete", False) else "plan"


def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("evaluate", evaluate_node)
    
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "rerank")
    graph.add_edge("rerank", "fetch")
    graph.add_edge("fetch", "synthesize")
    graph.add_edge("synthesize", "evaluate")
    
    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "plan": "plan",
            "end": END
        }
    )
    
    return graph.compile()
