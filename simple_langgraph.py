from typing import TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The messages in the conversation"]
    next: Annotated[str, "The next action to take"]

# Define the addition tool
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

# Initialize the tools
search_tool = TavilySearchResults()
tools = [search_tool, add_numbers]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define the agent
def agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response], "next": "tools"}

# Define the tools node
tools_node = ToolNode(tools)

# Define the router
def router(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", tools_node)

# Add edges
workflow.add_edge("agent", "tools")
workflow.add_edge("tools", "agent")

# Set entry point
workflow.set_entry_point("agent")

# Compile the graph
app = workflow.compile()

# Example usage
if __name__ == "__main__":
    # Sample query
    query = """What's the distance between Singapore and Shanghai, and between Shanghai and Beijing? 
    What's the total distance?"""
    
    # Initialize the state
    state = {
        "messages": [HumanMessage(content=query)],
        "next": "agent"
    }
    
    # Run the workflow
    for output in app.stream(state):
        if "messages" in output:
            last_message = output["messages"][-1]
            if isinstance(last_message, AIMessage):
                print(f"Assistant: {last_message.content}")
