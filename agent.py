from typing import TypedDict, List, Dict,Annotated
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langgraph.types import Send
from ddgs import DDGS
from langgraph.checkpoint.sqlite import SqliteSaver
import json
import operator


llm = ChatOllama(model="qwen2.5:7b")

class AgentState(TypedDict):
    goal:                str       # the user's original goal
    plan:                List[Dict] # list of steps [{id, task, status}]
    current_step_index:  int        # which step we're on
    findings:            Dict       # results from each step
    failed_steps:        List[str]  # steps that failed
    final_report:        str        # the synthesized answer
    companies_to_research: List[str]
    parallel_findings:   Annotated[Dict, operator.or_]  # merges parallel results


def plan_node(state: AgentState) -> dict:
    goal = state["goal"]
    
    response = llm.invoke(f"""
        You are a planning assistant. Break this goal into specific executable steps.
        Return ONLY a valid JSON array, nothing else. No explanation, no markdown.

        Each step must have:
        "id": step number (1, 2, 3...)
        "task": clear description of what to do
        "status": "pending"

        Goal: {goal}

        Example format:
        [
        {{"id": 1, "task": "search for top AI companies", "status": "pending"}},
        {{"id": 2, "task": "research OpenAI products", "status": "pending"}}
        ]
    """)

    raw = response.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        plan = json.loads(raw)
        print(f"\n  Plan created with {len(plan)} steps:")
        for step in plan:
            print(f"    {step['id']}. {step['task']}")
    except json.JSONDecodeError:
        print("  Failed to parse plan, using default")
        plan = [{"id": 1, "task": goal, "status": "pending"}]

    return {
        "plan": plan,
        "current_step_index": 0,
        "findings": {},
        "failed_steps": [],
        "final_report": ""
    }


def web_search(query: str) -> str:
    """Search the web for information"""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        output = ""
        for r in results:
            output += f"Title: {r['title']}\n{r['body']}\n\n"
        return output
    except Exception as e:
        return f"Search failed: {str(e)}"


def execute_node(state: AgentState) -> dict:
    plan = state["plan"]
    index = state["current_step_index"]
    findings = dict(state["findings"])
    failed_steps = list(state["failed_steps"])

    # check if all steps are done
    if index >= len(plan):
        return {"current_step_index": index}

    current_step = plan[index]
    task = current_step["task"]
    step_id = str(current_step["id"])

    print(f"\n  Executing step {current_step['id']}: {task}")

    # build context from previous findings
    context = ""
    if findings:
        context = "Previous findings:\n"
        for k, v in findings.items():
            context += f"Step {k}: {v[:200]}...\n"

    # ask LLM what to search for
    search_query_response = llm.invoke(
        f"Given this task: '{task}'\n"
        f"{context}\n"
        f"What is the best web search query to complete this task? "
        f"Return ONLY the search query, nothing else."
    )
    search_query = search_query_response.content.strip()
    print(f"  Searching: '{search_query}'")

    # execute the search
    search_result = web_search(search_query)

    # ask LLM to extract key info from results
    summary_response = llm.invoke(
        f"Task: {task}\n\n"
        f"Search results:\n{search_result}\n\n"
        f"Summarize the key information relevant to the task in 2-3 sentences."
    )
    summary = summary_response.content.strip()

    # determine if step succeeded
    if "No results" in search_result or "failed" in search_result.lower():
        print(f"  Step {current_step['id']} FAILED")
        failed_steps.append(step_id)
        findings[step_id] = "Data unavailable"
        plan[index]["status"] = "failed"
    else:   
        print(f"  Step {current_step['id']} succeeded")
        findings[step_id] = summary
        plan[index]["status"] = "completed"

    return {
        "plan": plan,
        "findings": findings,
        "failed_steps": failed_steps,
        "current_step_index": index + 1,
    }


def should_continue(state: AgentState) -> str:
    index = state["current_step_index"]
    plan = state["plan"]

    # all steps done → synthesize
    if index >= len(plan):
        return "synthesize"

    # too many failures → synthesize with what we have
    if len(state["failed_steps"]) > len(plan) // 2:
        print("\n  Too many failures, synthesizing with available data")
        return "synthesize"

    # more steps to do → execute next
    return "execute"


def synthesize_node(state: AgentState) -> dict:
    goal = state["goal"]
    findings = state["findings"]
    failed_steps = state["failed_steps"]

    # build a readable summary of all findings
    findings_text = ""
    for step_id, finding in findings.items():
        if finding != "Data unavailable":
            findings_text += f"Finding {step_id}: {finding}\n\n"

    if failed_steps:
        findings_text += f"\nNote: Steps {failed_steps} failed — data unavailable for those."

    print("\n  Synthesizing final report...")

    response = llm.invoke(
        f"Goal: {goal}\n\n"
        f"Research findings:\n{findings_text}\n\n"
        f"Write a clear, structured report answering the goal. "
        f"Include all key findings. Be concise but complete."
    )

    return {"final_report": response.content}

def route_to_parallel_research(state: AgentState):
    # Fan out
    companies = state.get("companies_to_research", [])
    return [Send("research_one", {"company": c, "goal": state["goal"]}) for c in companies]

def research_one_node(state: dict) -> dict:
    # Research a single company 
    company = state["company"]
    print(f"\n  [Parallel] Researching: {company}")
    
    result = web_search(f"{company} main products 2026")
    summary_response = llm.invoke(
        f"Summarize {company}'s main products in 2-3 sentences based on:\n{result}"
    )
    
    return {"parallel_findings": {company: summary_response.content.strip()}}


def prepare_parallel_node(state: AgentState) -> dict:
    """Ask the LLM to extract the list of items to research in parallel"""
    goal = state["goal"]
    response = llm.invoke(
        f"Read this goal and extract ONLY the specific company or product names "
        f"that are EXPLICITLY MENTIONED by name. Do not infer, guess, or add "
        f"any names that are not directly stated in the goal.\n\n"
        f"Goal: {goal}\n\n"
        f"Return ONLY a JSON array of the exact names mentioned. "
        f"If 3 companies are named, return exactly 3 items — no more, no less.\n\n"
        f"Example:\n"
        f'Goal: "compare OpenAI, Google, and Anthropic"\n'
        f'Output: ["OpenAI", "Google", "Anthropic"]'
    )
    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        companies = json.loads(raw)
    except json.JSONDecodeError:
        companies = []
    
    print(f"\n  Items to research in parallel: {companies}")
    return {"companies_to_research": companies}


def combine_parallel_node(state: AgentState) -> dict:
    """Combine all parallel findings into a final report"""
    findings = state.get("parallel_findings", {})
    goal = state["goal"]
    
    findings_text = "\n\n".join([f"{k}: {v}" for k, v in findings.items()])
    
    response = llm.invoke(
        f"Goal: {goal}\n\nFindings:\n{findings_text}\n\n"
        f"Write a structured comparison report."
    )
    return {"final_report": response.content}


# build the PARALLEL graph
parallel_graph = StateGraph(AgentState)
parallel_graph.add_node("prepare", prepare_parallel_node)
parallel_graph.add_node("research_one", research_one_node)
parallel_graph.add_node("combine", combine_parallel_node)

parallel_graph.set_entry_point("prepare")
parallel_graph.add_conditional_edges("prepare", route_to_parallel_research, ["research_one"])
parallel_graph.add_edge("research_one", "combine")
parallel_graph.add_edge("combine", END)

parallel_app = parallel_graph.compile()


sequential_graph = StateGraph(AgentState)
sequential_graph.add_node("plan", plan_node)
sequential_graph.add_node("execute", execute_node)
sequential_graph.add_node("synthesize", synthesize_node)

sequential_graph.set_entry_point("plan")
sequential_graph.add_edge("plan", "execute")
sequential_graph.add_conditional_edges("execute", should_continue, {
    "execute": "execute",
    "synthesize": "synthesize"
})
sequential_graph.add_edge("synthesize", END)

if __name__ == "__main__":
    import time

    mode = input("Mode — (1) sequential planning or (2) parallel research? ")
    goal = input("What is your goal? ")

    

    if mode == "1":
        with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
            app = sequential_graph.compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "session-1"}}

            result = app.invoke({
                "goal": goal,
                "plan": [],
                "current_step_index": 0,
                "findings": {},
                "failed_steps": [],
                "final_report": ""
            }, config=config)

            print("\n" + "="*50)
            print("FINAL REPORT")
            print("="*50)
            print(result["final_report"])
            print(f"\nFailed steps: {result['failed_steps']}")

    else:
        start = time.time()
        result = parallel_app.invoke({
            "goal": goal,
            "plan": [],
            "current_step_index": 0,
            "findings": {},
            "failed_steps": [],
            "final_report": "",
            "companies_to_research": [],
            "parallel_findings": {}
        })
        elapsed = time.time() - start

        print(f"\n{'='*50}")
        print(f"COMPLETED IN {elapsed:.1f} SECONDS")
        print(f"{'='*50}")
        print(result["final_report"])