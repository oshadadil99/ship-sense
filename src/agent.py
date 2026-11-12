"""LangGraph agent: agent node (LLM + tools) <-> tools node, with memory per thread_id.
Run:  python -m src.agent   (chat in the terminal; the portal should be running for live tracking)"""
import json
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.llm import get_chat_model
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are ShipSense, an assistant for a freight forwarder in Colombo, Sri Lanka.
Rules:
- Use the tools for every fact about shipments, carriers, documents and policies. Never invent numbers,
  dates, charges or container numbers.
- When you use a policy, cite its id in square brackets, e.g. [demurrage_and_detention#2].
- For charges, show the calculation step by step.
- A shipment is delayed when it is more than 48 hours late. Mention live portal data when you have it.
- If a tool returns an error, explain it simply and suggest what the user can do.
- If two sources disagree (e.g. a document vs the ERP shipment record), point out the mismatch clearly
  instead of picking one silently.
- Be short and clear."""

# StructuredTool reads each function's name, type hints and docstring -> the tool schema the LLM sees.
TOOLS = [StructuredTool.from_function(f) for f in ALL_TOOLS]


@lru_cache
def build_agent():
    llm = get_chat_model().bind_tools(TOOLS)  # the LLM now knows the 6 tools and can request them

    def agent_node(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}  # either an answer or a tool request

    graph = StateGraph(MessagesState)                  # state = the conversation messages
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))           # runs whatever tools the LLM asked for
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool calls? -> "tools", else -> END
    graph.add_edge("tools", "agent")                   # after tools, let the LLM think again
    return graph.compile(checkpointer=InMemorySaver())  # memory per thread_id (in RAM)


def chat(message: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}  # guardrail vs loops
    before = len(build_agent().get_state(config).values.get("messages", []))
    result = build_agent().invoke({"messages": [HumanMessage(message)]}, config)
    new = result["messages"][before:]  # only this turn's messages
    tools_used = [tc["name"] for m in new if isinstance(m, AIMessage) for tc in m.tool_calls]
    return {"answer": result["messages"][-1].content, "tools_used": tools_used}


if __name__ == "__main__":
    print("ShipSense agent - type 'quit' to exit, 'new' for a fresh conversation.")
    thread = 1
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "new":
            thread += 1
            print("(new conversation)")
            continue
        if q:
            out = chat(q, thread_id=str(thread))
            print(f"\nshipsense> {out['answer']}\n   [tools: {', '.join(out['tools_used']) or 'none'}]")
