# resume.py
from agent import graph, AgentState
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "session-1"}}

    # check what's saved so far
    state = app.get_state(config)
    print("Steps completed so far:", state.values.get("current_step_index"))
    print("Findings so far:", list(state.values.get("findings", {}).keys()))

    # resume from where it left off — pass None to continue
    result = app.invoke(None, config=config)

    print("\nFINAL REPORT:")
    print(result["final_report"])