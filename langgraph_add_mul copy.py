from typing import TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# Define the state type
class State(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The messages in the conversation"]
    next: Annotated[str, "The next action to take"]

# Define the tools
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

# Create the model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Create the tool node
tools = [add_numbers, multiply_numbers]
tool_node = ToolNode(tools)

# Define the router
def router(state: State) -> str:
    """Route to the next node based on the model's decision."""
    messages = state["messages"]
    response = model.invoke(messages)
    return "tools" if "tool" in response.content.lower() else "end"

# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("tools", tool_node)
workflow.add_node("model", model)

# Add edges
workflow.add_edge("model", router)
workflow.add_edge("tools", "model")

# Set entry point
workflow.set_entry_point("model")

# Compile the graph
app = workflow.compile()

# Example usage
if __name__ == "__main__":
    # Initialize the state
    state = {
        "messages": [HumanMessage(content="Add 5 and 3, then multiply the result by 2")],
        "next": "model"
    }
    
    # Run the graph
    result = app.invoke(state)
    print("Final result:", result["messages"][-1].content)
