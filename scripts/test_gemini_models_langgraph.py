import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Define the state for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

def test_model_with_langgraph(model_name: str):
    print(f"\n{'='*50}")
    print(f"Testing model: {model_name}")
    print(f"{'='*50}")

    # Initialize the Gemini model
    # ChatGoogleGenerativeAI automatically uses the GOOGLE_API_KEY environment variable
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
    )

    # Define the node function that calls the model
    def call_model(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # Build the simple graph
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    # Compile the graph
    app = workflow.compile()

    # Test the graph
    test_message = HumanMessage(content="Hello! Please reply with exactly 'API is working' and nothing else.")
    print(f"Sending message to LangGraph node using {model_name}...")

    try:
        result = app.invoke({"messages": [test_message]})
        final_message = result["messages"][-1].content
        print(f"✅ Success! Received response: {final_message}")
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")

if __name__ == "__main__":
    # Check if API keys are present (langchain_google_genai usually looks for GOOGLE_API_KEY)
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("Warning: Neither GOOGLE_API_KEY nor GEMINI_API_KEY environment variable is set.")
        print("The API calls will likely fail.")

    # Models to test
    models_to_test = [
        # "gemma-4-31b-it",
        # "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
        # "gemini-3.5-flash",
    ]

    for model in models_to_test:
        test_model_with_langgraph(model)
